import torch
import open3d as o3d
import numpy as np
from mc_mixin import _load_mc_ext

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
RESOLUTIONS = [32, 64, 128, 256]
N_RUNS = 20

SHOW_NORMALS = True
SHOW_MASK = False  # True = mostra solo le mesh con mask, False = mostra tutte

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------
def make_sphere(res):
    r = torch.arange(res, dtype=torch.float32)
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    c = res / 2
    sdf = ((x - c)**2 + (y - c)**2 + (z - c)**2).sqrt() - res * 0.3
    return sdf.cuda()

def make_mask(res):
    # crea uno spicchio: solo un quarto del cubo è True
    m = torch.zeros(res, res, res, dtype=torch.bool, device="cuda")
    m[res//2:, res//2:, :] = True
    return m

# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------
def time_cuda(fn):
    start = torch.cuda.Event(True)
    end   = torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(N_RUNS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_RUNS

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def run(mc):
    results = []
    for i, res in enumerate(RESOLUTIONS[:4]):  # solo 4 sfere
        grid = make_sphere(res)
        mask = make_mask(res) if SHOW_MASK else torch.Tensor()

        # warmup
        for _ in range(5):
            mc.marching_cubes(grid, 0.0, mask, compute_normals=True)
        torch.cuda.synchronize()

        out = [None]
        def fn():
            out[0] = mc.marching_cubes(grid, 0.0, mask, compute_normals=True)
        ms = time_cuda(fn)
        v, f, n = out[0]

        print(f"{res}³ | {'MASK' if SHOW_MASK else 'FULL':4} | {ms:.2f} ms | {v.shape[0]} verts | {f.shape[0]} faces")

        results.append({
            "res": res,
            "mask": SHOW_MASK,
            "v": v.cpu(),
            "f": f.cpu(),
            "n": n.cpu()
        })
    return results

# ------------------------------------------------------------
# Normals
# ------------------------------------------------------------
def get_normals_lineset(points, normals, length=0.05):
    pts = points.numpy()
    nrm = normals.numpy()
    line_pts = np.concatenate([pts, pts + nrm * length], axis=0)
    lines = [[i, i + len(pts)] for i in range(len(pts))]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(line_pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    return ls

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def show(results):
    geometries = []

    # 4 colori fissi
    colors_map = [
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
    ]

    offset = 0.0
    for i, rec in enumerate(results):
        v = rec["v"]
        f = rec["f"]
        n = rec["n"]

        if v.shape[0] == 0:
            continue

        verts = v.numpy()
        faces = f.numpy()
        norms = n.numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # normalizza tutto a unit cube
        bb = mesh.get_axis_aligned_bounding_box()
        scale = 1.0 / max(bb.get_extent())
        center = bb.get_center()
        verts = (verts - center) * scale
        norms = norms / np.linalg.norm(norms, axis=1, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms)

        # colore fisso
        mesh.paint_uniform_color(colors_map[i % 4])

        # shift per layout
        shift = np.array([offset, 0, 0])
        mesh.translate(shift)
        geometries.append(mesh)

        # normali
        if SHOW_NORMALS:
            ls = get_normals_lineset(torch.from_numpy(verts + shift), torch.from_numpy(norms))
            geometries.append(ls)

        offset += 2.0

    o3d.visualization.draw_geometries(geometries, zoom=0.4)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(torch.cuda.get_device_name(0))
    mc = _load_mc_ext()
    results = run(mc)
    print("Showing meshes...")
    show(results)