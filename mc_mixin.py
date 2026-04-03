from pathlib import Path
from torch.utils.cpp_extension import load
def _load_mc_ext():
    src_dir = Path(__file__).parent
    cu_file = src_dir / "marching_cubes.cu"

    print("Loading marching cubes CUDA extension from ", cu_file)
    
    if not cu_file.exists():
        raise FileNotFoundError(
            "marching_cubes.cu not found"
        )
    print("Compiling marching cubes CUDA extension...")
    return load(
        name="marching_cubes_gpu",
        sources=[str(cu_file)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3", "-std=c++17"],
    )

# Use:
# verts, faces = _mc_cuda.marching_cubes(grid, 0.0, mask)