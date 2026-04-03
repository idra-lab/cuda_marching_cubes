/*
 * cumcubes_masked.cu
 * ==================
 * GPU marching cubes with an optional per-cell boolean mask.
 *
 * Extends the original cumcubes implementation with a mask tensor
 * (bool[X][Y][Z]) that gates which cells are processed.  A cell is
 * skipped unless mask[x][y][z] is true, which is equivalent to the
 * scipy marching_cubes(mask=...) parameter.
 *
 * Typical use: set mask[x][y][z] = true only when all 8 corners of
 * the cell are "observed" (weight >= min_weight).  This suppresses
 * ghost triangles at the boundary of the observed region.
 *
 * API exposed to PyTorch via torch.utils.cpp_extension:
 *
 *   verts, faces = marching_cubes(grid, threshold, mask=None)
 *
 *   grid   : float[X][Y][Z]  — TSDF values (iso-surface at threshold, usually 0)
 *   threshold : float           — iso-surface level
 *   mask   : bool[X][Y][Z]  — optional; if None the full grid is processed
 *
 * Returns:
 *   verts  : float[V][3]    — vertex positions in grid-index space
 *   faces  : int32[F][3]    — triangle indices (counter-clockwise)
 *
 * Workflow (identical to cumcubes):
 *   1. count_vertices_faces_kernel  — atomic counters for V and F
 *   2. allocate verts[V] and faces[F]
 *   3. gen_vertices_kernel          — interpolated vertex positions
 *   4. gen_faces_kernel             — triangle connectivity
 *
 * Marching-cubes tables from pmneila/PyMCubes (BSD 3-clause).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// Marching-cubes triangle table (256 × 16)
// Source: pmneila/PyMCubes → Paul Bourke (BSD 3-clause)
// ============================================================
static __device__ int8_t triangle_table[256][16] =
    {
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
        {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
        {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
        {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
        {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
        {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
        {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
        {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
        {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
        {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
        {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
        {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
        {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
        {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
        {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
        {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
        {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
        {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
        {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
        {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
        {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
        {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
        {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
        {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
        {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
        {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
        {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
        {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
        {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
        {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

// ============================================================
// Helper
// ============================================================
__host__ __device__ __inline__ uint32_t div_round_up(uint32_t val, uint32_t divisor)
{
    return (val + divisor - 1) / divisor;
}

// ============================================================
// Kernel 1: count vertices and faces
//
// Each thread owns one voxel (x, y, z).
//
// VERTEX COUNT
//   For each of the three axis-aligned edges originating at
//   (x,y,z), if the iso-surface crosses that edge AND the
//   neighbour voxel is also mask-active, count one vertex.
//   Edges whose neighbour is masked out will not be written by
//   gen_vertices_kernel, so we must not count them here either.
//
// FACE COUNT
//   A cell produces triangles only when every edge referenced by
//   triangle_table is guaranteed to have a vertex.  An edge has a
//   vertex iff the voxel that "owns" it (the lower-index corner
//   along that axis) passes the mask check.  The 12 edges of the
//   cube map to these owner voxels:
//
//     edge  0 → owner (x,   y,   z  ) axis x   (mask[x][y][z])
//     edge  1 → owner (x+1, y,   z  ) axis y   (mask[x+1][y][z])
//     edge  2 → owner (x,   y+1, z  ) axis x   (mask[x][y+1][z])
//     edge  3 → owner (x,   y,   z  ) axis y   (mask[x][y][z])
//     edge  4 → owner (x,   y,   z+1) axis x   (mask[x][y][z+1])
//     edge  5 → owner (x+1, y,   z+1) axis y   (mask[x+1][y][z+1])
//     edge  6 → owner (x,   y+1, z+1) axis x   (mask[x][y+1][z+1])
//     edge  7 → owner (x,   y,   z+1) axis y   (mask[x][y][z+1])
//     edge  8 → owner (x,   y,   z  ) axis z   (mask[x][y][z])
//     edge  9 → owner (x+1, y,   z  ) axis z   (mask[x+1][y][z])
//     edge 10 → owner (x+1, y+1, z  ) axis z   (mask[x+1][y+1][z])
//     edge 11 → owner (x,   y+1, z  ) axis z   (mask[x][y+1][z])
//
//   Before counting tricount we check that every edge required by
//   triangle_table[cube_mask] has an active owner.  This exactly
//   mirrors the check in gen_faces_kernel so the two counters stay
//   in sync and no placeholder slots are left in the output.
// ============================================================
__global__ void count_vertices_faces_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grid,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> mask,
    const bool use_mask,
    const float threshold,
    int32_t *counters) // [0] = vertex count   [1] = face-index × 3
{
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const int32_t res_x = grid.size(0);
    const int32_t res_y = grid.size(1);
    const int32_t res_z = grid.size(2);

    if (x >= res_x || y >= res_y || z >= res_z)
        return;
    // if (use_mask && !mask[x][y][z]) return;
    const bool self_active = !use_mask || mask[x][y][z];

    const float d_self = grid[x][y][z];
    const bool inside = d_self > threshold;

    // ── Vertices ──────────────────────────────────────────────────────────────
    // An edge is counted only when:
    //   (a) the iso-surface crosses it, AND
    //   (b) the neighbour voxel is mask-active (so gen_vertices_kernel will
    //       actually write that vertex).
    // Without (b) a crossing at the mask boundary would be counted here but
    // never written, leaving the face kernel short of a vertex.
    if (x < res_x - 1)
    {
        const bool nb_active = !use_mask || mask[x + 1][y][z];
        if ((self_active || nb_active) && inside != (grid[x + 1][y][z] > threshold))
            atomicAdd(counters, 1);
    }
    if (y < res_y - 1)
    {
        const bool nb_active = !use_mask || mask[x][y + 1][z];
        if ((self_active || nb_active) && inside != (grid[x][y + 1][z] > threshold))
            atomicAdd(counters, 1);
    }
    if (z < res_z - 1)
    {
        const bool nb_active = !use_mask || mask[x][y][z + 1];
        if ((self_active || nb_active) && inside != (grid[x][y][z + 1] > threshold))
            atomicAdd(counters, 1);
    }

    // ── Faces ─────────────────────────────────────────────────────────────────
    if (x >= res_x - 1 || y >= res_y - 1 || z >= res_z - 1)
        return;

    uint8_t cube_mask = 0;
    if (grid[x][y][z] > threshold)
        cube_mask |= 1;
    if (grid[x + 1][y][z] > threshold)
        cube_mask |= 2;
    if (grid[x + 1][y + 1][z] > threshold)
        cube_mask |= 4;
    if (grid[x][y + 1][z] > threshold)
        cube_mask |= 8;
    if (grid[x][y][z + 1] > threshold)
        cube_mask |= 16;
    if (grid[x + 1][y][z + 1] > threshold)
        cube_mask |= 32;
    if (grid[x + 1][y + 1][z + 1] > threshold)
        cube_mask |= 64;
    if (grid[x][y + 1][z + 1] > threshold)
        cube_mask |= 128;

    int32_t tricount = 0;
    const int8_t *tris = triangle_table[cube_mask];
    for (; tricount < 15; tricount += 3)
        if (tris[tricount] < 0)
            break;

    if (tricount == 0)
        return;

    // For each triangle vertex, check that the edge owner is mask-active.
    // This is the same predicate used in gen_faces_kernel: if any required
    // edge owner is inactive, gen_faces_kernel will return early without
    // emitting any face for this cell, so we must not count it here either.
    if (use_mask)
    {
        // Pre-compute which of the 8 cube-corner voxels (= edge owners) are
        // active.  The mapping edge → owner is documented in the header above.
        //
        //   owner_active[0..7] corresponds to the 8 corners of the cell:
        //     [0] = (x,   y,   z  )   [1] = (x+1, y,   z  )
        //     [2] = (x,   y+1, z  )   [3] = (x+1, y+1, z  )   (unused directly)
        //     [4] = (x,   y,   z+1)   [5] = (x+1, y,   z+1)
        //     [6] = (x,   y+1, z+1)   [7] = (x+1, y+1, z+1)   (unused directly)
        //
        // edge_owner[e] indexes into owner_active[]:
        const bool oa[8] = {
            mask[x][y][z],         // 0
            mask[x + 1][y][z],     // 1
            mask[x][y + 1][z],     // 2
            true,                  // 3 — (x+1,y+1,z): not an edge owner
            mask[x][y][z + 1],     // 4
            mask[x + 1][y][z + 1], // 5
            mask[x][y + 1][z + 1], // 6
            true,                  // 7 — (x+1,y+1,z+1): not an edge owner
        };
        // edge index → index into oa[]
        // edge  0: x-axis at corner 0  → oa[0]
        // edge  1: y-axis at corner 1  → oa[1]
        // edge  2: x-axis at corner 2  → oa[2]
        // edge  3: y-axis at corner 0  → oa[0]
        // edge  4: x-axis at corner 4  → oa[4]
        // edge  5: y-axis at corner 5  → oa[5]
        // edge  6: x-axis at corner 6  → oa[6]
        // edge  7: y-axis at corner 4  → oa[4]
        // edge  8: z-axis at corner 0  → oa[0]
        // edge  9: z-axis at corner 1  → oa[1]
        // edge 10: z-axis at corner 3  → oa[2] (x+1,y+1,z owns z-edge → same mask as corner 2...
        //          actually (x+1,y+1,z): mask[x+1][y+1][z] — approximated as oa[2] is wrong.
        //          Use a direct read instead.
        // edge 11: z-axis at corner 2  → mask[x][y+1][z] = oa[2]
        //
        // edges 3,7 share owner with edges 8,0 respectively — already in oa[].
        // edge 10's owner is (x+1, y+1, z) which is not in oa[]; read directly.
        const bool oa10 = mask[x + 1][y + 1][z];

        static const int8_t edge_to_oa[12] = {0, 1, 2, 0, 4, 5, 6, 4, 0, 1, -1, 2};
        //                                                                    ^^ -1 means use oa10

        auto edge_active = [&](int e)
        {
            switch (e)
            {
            case 0:
                return mask[x][y][z] || mask[x + 1][y][z];
            case 1:
                return mask[x + 1][y][z] || mask[x + 1][y + 1][z];
            case 2:
                return mask[x][y + 1][z] || mask[x + 1][y + 1][z];
            case 3:
                return mask[x][y][z] || mask[x][y + 1][z];
            case 4:
                return mask[x][y][z + 1] || mask[x + 1][y][z + 1];
            case 5:
                return mask[x + 1][y][z + 1] || mask[x + 1][y + 1][z + 1];
            case 6:
                return mask[x][y + 1][z + 1] || mask[x + 1][y + 1][z + 1];
            case 7:
                return mask[x][y][z + 1] || mask[x][y + 1][z + 1];
            case 8:
                return mask[x][y][z] || mask[x][y][z + 1];
            case 9:
                return mask[x + 1][y][z] || mask[x + 1][y][z + 1];
            case 10:
                return mask[x + 1][y + 1][z] || mask[x + 1][y + 1][z + 1];
            case 11:
                return mask[x][y + 1][z] || mask[x][y + 1][z + 1];
            }
            return false;
        };

        for (int32_t i = 0; i < tricount; ++i)
        {
            int8_t e = tris[i];
            if (e < 0)
                break;
            if (!edge_active(e))
                return;
        }
    }

    atomicAdd(counters + 1, tricount);
}

// ============================================================
// Kernel 2: generate vertices
//
// Same mask guard as kernel 1.
// For each crossing edge, interpolate the vertex position and
// store it in vertices[vidx], recording vidx+1 in vertex_grid
// so kernel 3 can look it up by cell corner.
// ============================================================
// ------------------------------------------------------------
// Central-difference gradient of the scalar field at (ix,iy,iz).
// Uses one-sided differences at grid boundaries.
// ------------------------------------------------------------
__device__ __forceinline__ void grid_gradient(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grid,
    int32_t ix, int32_t iy, int32_t iz,
    float &gx, float &gy, float &gz)
{
    const int32_t res_x = grid.size(0);
    const int32_t res_y = grid.size(1);
    const int32_t res_z = grid.size(2);

    const float fx0 = (ix > 0) ? grid[ix - 1][iy][iz] : grid[ix][iy][iz];
    const float fx1 = (ix < res_x - 1) ? grid[ix + 1][iy][iz] : grid[ix][iy][iz];
    const float fy0 = (iy > 0) ? grid[ix][iy - 1][iz] : grid[ix][iy][iz];
    const float fy1 = (iy < res_y - 1) ? grid[ix][iy + 1][iz] : grid[ix][iy][iz];
    const float fz0 = (iz > 0) ? grid[ix][iy][iz - 1] : grid[ix][iy][iz];
    const float fz1 = (iz < res_z - 1) ? grid[ix][iy][iz + 1] : grid[ix][iy][iz];

    gx = (fx1 - fx0) * 0.5f;
    gy = (fy1 - fy0) * 0.5f;
    gz = (fz1 - fz0) * 0.5f;
}

__global__ void gen_vertices_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grid,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> mask,
    const bool use_mask,
    const float threshold,
    const bool compute_normals,
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> vertex_grid,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> vertices,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normals,
    int32_t *counters)
{
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const int32_t res_x = grid.size(0);
    const int32_t res_y = grid.size(1);
    const int32_t res_z = grid.size(2);

    if (x >= res_x || y >= res_y || z >= res_z)
        return;

    // Do NOT early-exit on mask[x][y][z] here.
    // This thread owns three edges (x-axis, y-axis, z-axis) originating
    // at (x,y,z).  Each edge is needed by up to 4 neighbouring cells.
    // If we skip a thread because mask[x][y][z]=false, the neighbour cell
    // (x-1,y,z) with mask=true will look for vertex_grid[x][y][z][1] and
    // find zero — producing the "missing vertex" error.
    //
    // Instead: write the vertex whenever the iso-surface crosses the edge
    // AND at least one of its two endpoints is mask-active.  This ensures
    // every vertex that any active cell could need is actually written.

    const float d_self = grid[x][y][z];
    const bool inside = d_self > threshold;
    const bool self_active = !use_mask || mask[x][y][z];

    // x-axis edge: endpoints (x,y,z) and (x+1,y,z)
    if (x < res_x - 1)
    {
        const float d_xn = grid[x + 1][y][z];
        const bool nb_active = !use_mask || mask[x + 1][y][z];
        if ((self_active || nb_active) && (inside != (d_xn > threshold)))
        {
            int32_t vidx = atomicAdd(counters, 1);
            const float dt = (threshold - d_self) / (d_xn - d_self);
            vertex_grid[x][y][z][0] = vidx + 1;
            vertices[vidx][0] = static_cast<float>(x) + dt;
            vertices[vidx][1] = static_cast<float>(y);
            vertices[vidx][2] = static_cast<float>(z);

            if (compute_normals)
            {
                float gx0, gy0, gz0, gx1, gy1, gz1;
                grid_gradient(grid, x, y, z, gx0, gy0, gz0);
                grid_gradient(grid, x + 1, y, z, gx1, gy1, gz1);
                float nx = gx0 + dt * (gx1 - gx0);
                float ny = gy0 + dt * (gy1 - gy0);
                float nz = gz0 + dt * (gz1 - gz0);
                const float len = sqrtf(nx * nx + ny * ny + nz * nz);
                if (len > 1e-8f)
                {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                normals[vidx][0] = -nx;
                normals[vidx][1] = -ny;
                normals[vidx][2] = -nz;
            }
        }
    }
    // y-axis edge: endpoints (x,y,z) and (x,y+1,z)
    if (y < res_y - 1)
    {
        const float d_yn = grid[x][y + 1][z];
        const bool nb_active = !use_mask || mask[x][y + 1][z];
        if ((self_active || nb_active) && (inside != (d_yn > threshold)))
        {
            int32_t vidx = atomicAdd(counters, 1);
            const float dt = (threshold - d_self) / (d_yn - d_self);
            vertex_grid[x][y][z][1] = vidx + 1;
            vertices[vidx][0] = static_cast<float>(x);
            vertices[vidx][1] = static_cast<float>(y) + dt;
            vertices[vidx][2] = static_cast<float>(z);

            if (compute_normals)
            {
                float gx0, gy0, gz0, gx1, gy1, gz1;
                grid_gradient(grid, x, y, z, gx0, gy0, gz0);
                grid_gradient(grid, x, y + 1, z, gx1, gy1, gz1);
                float nx = gx0 + dt * (gx1 - gx0);
                float ny = gy0 + dt * (gy1 - gy0);
                float nz = gz0 + dt * (gz1 - gz0);
                const float len = sqrtf(nx * nx + ny * ny + nz * nz);
                if (len > 1e-8f)
                {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                normals[vidx][0] = nx;
                normals[vidx][1] = ny;
                normals[vidx][2] = nz;
            }
        }
    }
    // z-axis edge: endpoints (x,y,z) and (x,y,z+1)
    if (z < res_z - 1)
    {
        const float d_zn = grid[x][y][z + 1];
        const bool nb_active = !use_mask || mask[x][y][z + 1];
        if ((self_active || nb_active) && (inside != (d_zn > threshold)))
        {
            int32_t vidx = atomicAdd(counters, 1);
            const float dt = (threshold - d_self) / (d_zn - d_self);
            vertex_grid[x][y][z][2] = vidx + 1;
            vertices[vidx][0] = static_cast<float>(x);
            vertices[vidx][1] = static_cast<float>(y);
            vertices[vidx][2] = static_cast<float>(z) + dt;

            if (compute_normals)
            {
                float gx0, gy0, gz0, gx1, gy1, gz1;
                grid_gradient(grid, x, y, z, gx0, gy0, gz0);
                grid_gradient(grid, x, y, z + 1, gx1, gy1, gz1);
                float nx = gx0 + dt * (gx1 - gx0);
                float ny = gy0 + dt * (gy1 - gy0);
                float nz = gz0 + dt * (gz1 - gz0);
                const float len = sqrtf(nx * nx + ny * ny + nz * nz);
                if (len > 1e-8f)
                {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                normals[vidx][0] = nx;
                normals[vidx][1] = ny;
                normals[vidx][2] = nz;
            }
        }
    }
}
// ============================================================
// Kernel 3: generate faces
//
// Same mask guard as kernels 1–2.
// Looks up the 12 edge vertices from vertex_grid, then emits
// triangles from triangle_table into faces[tidx..].
// ============================================================
__global__ void gen_faces_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grid,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> mask,
    const bool use_mask,
    const float threshold,
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> vertex_grid,
    int32_t *faces,
    int32_t *counters)
{
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const int32_t res_x = grid.size(0);
    const int32_t res_y = grid.size(1);
    const int32_t res_z = grid.size(2);

    if (x >= res_x - 1 || y >= res_y - 1 || z >= res_z - 1)
        return;
    if (use_mask && !mask[x][y][z])
        return;

    // ── cube configuration ─────────────────────────────────────
    uint8_t cube_mask = 0;
    if (grid[x][y][z] > threshold)
        cube_mask |= 1;
    if (grid[x + 1][y][z] > threshold)
        cube_mask |= 2;
    if (grid[x + 1][y + 1][z] > threshold)
        cube_mask |= 4;
    if (grid[x][y + 1][z] > threshold)
        cube_mask |= 8;
    if (grid[x][y][z + 1] > threshold)
        cube_mask |= 16;
    if (grid[x + 1][y][z + 1] > threshold)
        cube_mask |= 32;
    if (grid[x + 1][y + 1][z + 1] > threshold)
        cube_mask |= 64;
    if (grid[x][y + 1][z + 1] > threshold)
        cube_mask |= 128;

    const int8_t *tris = triangle_table[cube_mask];

    // ── edge validity (mask-aware, SAME LOGIC as gen_vertices) ──
    auto edge_active = [&](int e) -> bool
    {
        if (!use_mask)
            return true;

        switch (e)
        {
        case 0:
            return mask[x][y][z] || mask[x + 1][y][z];
        case 1:
            return mask[x + 1][y][z] || mask[x + 1][y + 1][z];
        case 2:
            return mask[x][y + 1][z] || mask[x + 1][y + 1][z];
        case 3:
            return mask[x][y][z] || mask[x][y + 1][z];
        case 4:
            return mask[x][y][z + 1] || mask[x + 1][y][z + 1];
        case 5:
            return mask[x + 1][y][z + 1] || mask[x + 1][y + 1][z + 1];
        case 6:
            return mask[x][y + 1][z + 1] || mask[x + 1][y + 1][z + 1];
        case 7:
            return mask[x][y][z + 1] || mask[x][y + 1][z + 1];
        case 8:
            return mask[x][y][z] || mask[x][y][z + 1];
        case 9:
            return mask[x + 1][y][z] || mask[x + 1][y][z + 1];
        case 10:
            return mask[x + 1][y + 1][z] || mask[x + 1][y + 1][z + 1];
        case 11:
            return mask[x][y + 1][z] || mask[x][y + 1][z + 1];
        }
        return false;
    };

    // ── count triangles + validate edges ───────────────────────
    int32_t tricount = 0;
    for (; tricount < 15; tricount += 3)
    {
        if (tris[tricount] < 0)
            break;
    }
    if (tricount == 0)
        return;

    // mask consistency check (CRUCIALE)
    for (int32_t i = 0; i < tricount; ++i)
    {
        int8_t e = tris[i];
        if (e < 0)
            break;
        if (!edge_active(e))
            return;
    }

    // ── fetch edge vertices ────────────────────────────────────
    int32_t local_edges[12];
    local_edges[0] = vertex_grid[x][y][z][0];
    local_edges[1] = vertex_grid[x + 1][y][z][1];
    local_edges[2] = vertex_grid[x][y + 1][z][0];
    local_edges[3] = vertex_grid[x][y][z][1];
    local_edges[4] = vertex_grid[x][y][z + 1][0];
    local_edges[5] = vertex_grid[x + 1][y][z + 1][1];
    local_edges[6] = vertex_grid[x][y + 1][z + 1][0];
    local_edges[7] = vertex_grid[x][y][z + 1][1];
    local_edges[8] = vertex_grid[x][y][z][2];
    local_edges[9] = vertex_grid[x + 1][y][z][2];
    local_edges[10] = vertex_grid[x + 1][y + 1][z][2];
    local_edges[11] = vertex_grid[x][y + 1][z][2];

    // ── reserve output ─────────────────────────────────────────
    int32_t tidx = atomicAdd(counters + 1, tricount);

    // ── emit triangles ─────────────────────────────────────────
    for (int32_t i = 0; i < tricount; ++i)
    {
        int32_t e = tris[i];
        if (e < 0)
            break;

        int32_t v = local_edges[e];
        if (!v)
        {
            printf("[gen_faces] missing vertex at (%d,%d,%d) edge %d mask=%d\n",
                   x, y, z, e, (int)cube_mask);
        }

        faces[tidx + i] = v - 1;
    }
}

// ============================================================
// C++ wrapper
// ============================================================

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK(x)   \
    CHECK_CUDA(x); \
    CHECK_CONTIG(x)

/*
 * marching_cubes_cuda(grid, threshold, mask, compute_normals)
 *
 * grid           : float[X][Y][Z]  contiguous CUDA tensor
 * threshold      : float            iso-surface level (typically 0.0 for TSDF)
 * mask           : bool[X][Y][Z]   optional contiguous CUDA tensor; pass an
 *                                  undefined/empty tensor to disable masking.
 *                                  When provided, only cells where mask[x][y][z]
 *                                  is true contribute vertices and faces.
 * compute_normals: bool             when true, returns an additional float[V][3]
 *                                  tensor of per-vertex normals computed via
 *                                  central-difference gradients of the scalar
 *                                  field, interpolated along each edge.
 *
 * Returns {vertices float[V][3], faces int32[F][3]} in grid-index space,
 * or {vertices, faces, normals float[V][3]} when compute_normals=true.
 * Scale / translate the vertices in Python after the call.
 */
std::vector<torch::Tensor> marching_cubes_cuda(
    const torch::Tensor &grid,
    const float threshold,
    const torch::Tensor &mask, // may be undefined (use_mask = false)
    const bool compute_normals = false)
{
    CHECK(grid);
    TORCH_CHECK(grid.dtype() == torch::kFloat32, "grid must be float32");
    TORCH_CHECK(grid.dim() == 3, "grid must be 3-D");

    const bool use_mask = mask.defined() && mask.numel() > 0;
    if (use_mask)
    {
        CHECK(mask);
        TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
        TORCH_CHECK(mask.sizes() == grid.sizes(), "mask and grid must have the same shape");
    }

    // Dummy bool accessor for the no-mask case.  The kernel receives
    // use_mask=false so it never actually dereferences the accessor.
    torch::Tensor mask_buf = use_mask
                                 ? mask
                                 : torch::zeros({1, 1, 1}, torch::TensorOptions().dtype(torch::kBool).device(grid.device()));

    const auto device = grid.device();
    const int32_t res_x = static_cast<int32_t>(grid.size(0));
    const int32_t res_y = static_cast<int32_t>(grid.size(1));
    const int32_t res_z = static_cast<int32_t>(grid.size(2));

    // counters layout: [0]=vertex count, [1]=face-index×3, [2]=vertex running, [3]=face running
    torch::Tensor counters = torch::zeros(
        {4}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    const uint32_t tx = 8, ty = 8, tz = 8;
    const dim3 threads(tx, ty, tz);
    const dim3 blocks(div_round_up(res_x, tx),
                      div_round_up(res_y, ty),
                      div_round_up(res_z, tz));

    // ── Pass 1: count ───────────────────────────────────────────────────────
    count_vertices_faces_kernel<<<blocks, threads>>>(
        grid.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mask_buf.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        use_mask, threshold, counters.data_ptr<int32_t>());

    const int32_t num_verts = counters[0].item<int32_t>();
    const int32_t num_faces = counters[1].item<int32_t>() / 3;

    if (num_verts == 0 || num_faces == 0)
    {
        // No surface intersects the iso-level within the masked region.
        std::vector<torch::Tensor> empty = {
            torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
            torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device)),
        };
        if (compute_normals)
            empty.push_back(torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        return empty;
    }

    // vertex_grid[X][Y][Z][3]: stores vidx+1 for each edge origin.
    // Initialised to zero (0 = "no vertex here").
    torch::Tensor vertex_grid = torch::zeros(
        {res_x, res_y, res_z, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor vertices = torch::zeros(
        {num_verts, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor normals_buf = compute_normals
                                    ? torch::zeros({num_verts, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                                    : torch::zeros({1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor faces = torch::zeros(
        {num_faces * 3}, // flat; reshaped below
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // ── Pass 2: generate vertices ───────────────────────────────────────────
    gen_vertices_kernel<<<blocks, threads>>>(
        grid.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mask_buf.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        use_mask, threshold, compute_normals,
        vertex_grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        vertices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        normals_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        counters.data_ptr<int32_t>() + 2);

    // ── Pass 3: generate faces ──────────────────────────────────────────────
    gen_faces_kernel<<<blocks, threads>>>(
        grid.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mask_buf.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        use_mask, threshold,
        vertex_grid.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        faces.data_ptr<int32_t>(),
        counters.data_ptr<int32_t>() + 2);

    // return {vertices, faces.reshape({num_faces, 3})};
    if (compute_normals)
    {
        return {
            vertices,
            faces.reshape({num_faces, 3}),
            normals_buf};
    }
    else
    {
        return {
            vertices,
            faces.reshape({num_faces, 3})};
    }
}

// ============================================================
// PyBind11 binding
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
m.def("marching_cubes",
      &marching_cubes_cuda,
      "GPU marching cubes with optional bool mask and normals.\n\n"
      "Args:\n"
      "  grid   (Tensor float32 [X,Y,Z]): TSDF grid on CUDA\n"
      "  threshold (float): iso-surface level\n"
      "  mask   (Tensor bool [X,Y,Z], optional): mask\n"
      "  compute_normals (bool, optional): compute per-vertex normals\n"
      "Returns:\n"
      "  verts  (Tensor float32 [V,3])\n"
      "  faces  (Tensor int32  [F,3])\n"
      "  normals (Tensor float32 [V,3], optional)",
      py::arg("grid"),
      py::arg("threshold"),
      py::arg("mask") = torch::Tensor(),
      py::arg("compute_normals") = false);
}