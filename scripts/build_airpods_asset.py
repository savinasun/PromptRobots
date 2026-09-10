#!/usr/bin/env python
"""Rebuild astra_yam/assets/airpods/ from Apple's AR Quick Look model of the AirPods Pro.

Apple ships an exactly-dimensioned USDZ for AR Quick Look; it already separates the charging case
into a body group and a lid group, with the lid posed *open*.  This script

  1. downloads (or reads) the USDZ and pulls out the two groups,
  2. finds the lid's mating-rim plane and rotates the lid about the hinge until it is shut,
  3. re-expresses both meshes in SimCase's upright frame (x = depth, y = width, z = up, origin at the
     case centre), scaled so the closed case is exactly SimCase.H x SimCase.W x SimCase.D, and
  4. writes case_body.ply / case_lid.ply relative to SimCase.body_center() / SimCase.lid_center().

Needs `pip install usd-core` (conversion only - astra_yam itself just reads the PLYs via trimesh).

    python scripts/build_airpods_asset.py [--usdz PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

USDZ_URL = ("https://www.apple.com/105/media/us/airpods-pro/2019/"
            "1299e2f5_9206_4470_b28e_08307a42f19b/quicklook/airpods_pro_ios13.usdz")
# The USD prim names are opaque hashes; these two are the charging case (the others are the earbuds
# and a shadow plane, which the AR asset floats above the case and which we do not need).
BODY_PRIM, LID_PRIM = "zzBtSmxeDxAZRRE", "BoFsEvBGaAjKkRU"
APPLE_MM = {"W": 60.6, "D": 45.2, "H": 21.7}      # published AirPods Pro case width / height / depth


def load_groups(usdz: Path) -> dict[str, trimesh.Trimesh]:
    """Triangulated, world-space (metre) meshes for the case body and lid."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usdz))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parts: dict[str, list] = collections.defaultdict(list)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = prim.GetPath().pathString.strip("/").split("/")
        group = {BODY_PRIM: "body", LID_PRIM: "lid"}.get(path[1] if len(path) > 1 else "")
        if group is None:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        faces, off = [], 0
        for c in counts:                                            # fan-triangulate n-gons
            f = idx[off:off + c]
            off += c
            faces += [[f[0], f[i], f[i + 1]] for i in range(1, c - 1)]
        m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)   # USD is row-vector
        v = (pts @ m[:3, :3] + m[3, :3]) * mpu
        parts[group].append(trimesh.Trimesh(vertices=v, faces=np.asarray(faces), process=False))
    missing = {"body", "lid"} - set(parts)
    if missing:
        raise RuntimeError(f"USDZ did not contain {missing}; prim names may have changed")
    return {k: trimesh.util.concatenate(v) for k, v in parts.items()}


def rim_normal(lid: trimesh.Trimesh) -> np.ndarray:
    """Outward normal of the lid's mating rim: the largest planar face cluster parallel to the hinge."""
    n, area = lid.face_normals, lid.area_faces
    flat = np.abs(n[:, 0]) < 0.15                                   # faces parallel to the hinge axis (x)
    keys, inv = np.unique(np.round(n[flat], 1), axis=0, return_inverse=True)
    biggest = np.argmax(np.bincount(inv, weights=area[flat]))
    sel = np.flatnonzero(flat)[inv == biggest]
    v = (n[sel] * area[sel, None]).sum(0)
    return v / np.linalg.norm(v)


def close_lid(body: trimesh.Trimesh, lid: trimesh.Trimesh):
    """Rotation about the hinge (+ translation) that shuts the lid, and the hinge point it implies."""
    nrm = rim_normal(lid)
    # rotate about x so the rim normal points straight down; that is the shut lid
    theta = np.pi - np.arctan2(nrm[2], nrm[1])
    c, s = np.cos(theta), np.sin(theta)
    v = lid.vertices
    q = np.column_stack([v[:, 0], c * v[:, 1] - s * v[:, 2], s * v[:, 1] + c * v[:, 2]])
    b = body.vertices
    dz = ((b[:, 2].min() - q[:, 2].min()) + (b[:, 2].max() - q[:, 2].max())) / 2   # flush front/back
    dy = (b[:, 1].min() + APPLE_MM["D"] / 1000) - q[:, 1].max()                    # published total height
    rot2 = np.array([[c, -s], [s, c]])
    hinge_yz = np.linalg.solve(np.eye(2) - rot2, [dy, dz])          # fixed point of the shutting motion
    return theta, np.array([0.0, dy, dz]), np.array([0.0, *hinge_yz])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", type=Path, default=Path("astra_yam/assets/airpods/airpods_pro_ios13.usdz"))
    ap.add_argument("--out", type=Path, default=Path("astra_yam/assets/airpods"))
    args = ap.parse_args()

    if not args.usdz.exists():
        args.usdz.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {USDZ_URL}")
        req = urllib.request.Request(USDZ_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(args.usdz, "wb") as f:
            f.write(r.read())
    groups = load_groups(args.usdz)
    body, lid = groups["body"], groups["lid"]
    print(f"body {len(body.faces)} faces {np.round(body.extents * 1000, 2)} mm; "
          f"lid {len(lid.faces)} faces {np.round(lid.extents * 1000, 2)} mm (open)")

    theta, shift, hinge = close_lid(body, lid)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    lid_closed = lid.vertices @ rot.T + shift
    print(f"lid shut by {np.degrees(theta):.2f} deg about hinge (usd y,z) = {np.round(hinge[1:] * 1000, 3)} mm")

    lo = np.minimum(body.vertices.min(0), lid_closed.min(0))
    hi = np.maximum(body.vertices.max(0), lid_closed.max(0))
    centre = (lo + hi) / 2
    got = (hi - lo) * 1000                                          # USD axes: x width, y up, z depth
    want = np.array([APPLE_MM["W"], APPLE_MM["D"], APPLE_MM["H"]])
    print(f"closed case {np.round(got, 2)} mm vs Apple {want} mm")
    if np.abs(got - want).max() > 0.15:
        raise SystemExit(f"closed case {got} does not match Apple's published dimensions {want}")

    # USD (x=width, y=up, z=depth) -> SimCase upright frame (x=depth, y=width, z=up), exact scale
    perm = np.array([2, 0, 1])
    scale = want[[2, 0, 1]] / got[[2, 0, 1]]   # vertices are already metres
    def to_case(v):
        return (v - centre)[:, perm] * scale

    body_c, lid_c = to_case(body.vertices), to_case(lid_closed)
    hinge_c = to_case(hinge[None, :])[0]
    seam_z = body_c[:, 2].max()
    top_z = lid_c[:, 2].max()
    LID = top_z - seam_z
    print(f"\nSimCase constants derived from the model (metres):")
    print(f"  W, D, H  = {want[0] / 1000:.4f}, {want[1] / 1000:.4f}, {want[2] / 1000:.4f}")
    print(f"  LID      = {LID:.5f}   (visible lid height above the seam)")
    print(f"  HINGE    = ({hinge_c[0]:.5f}, 0.0, {hinge_c[2]:.5f})  relative to the case centre, upright")
    print(f"  lid flange reaches {(seam_z - lid_c[:, 2].min()) * 1000:.2f} mm below the seam")

    args.out.mkdir(parents=True, exist_ok=True)
    D, H = want[1] / 1000, want[2] / 1000
    for name, verts, mesh, origin in (("case_body", body_c, body, np.array([0, 0, -LID / 2])),
                                      ("case_lid", lid_c, lid, np.array([0, 0, D / 2 - LID / 2]))):
        out = trimesh.Trimesh(vertices=verts - origin, faces=mesh.faces, process=False)
        path = args.out / f"{name}.ply"
        path.write_bytes(trimesh.exchange.ply.export_ply(out, encoding="binary"))
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB, {len(out.faces)} faces)")


if __name__ == "__main__":
    main()
