"""Align the current scan to the baseline (ARCHITECTURE §6.3): point-to-plane ICP, then accept or reject.

numpy + scipy only. A fixed rig converges at ~identity; the same code path takes an initial pose for the moving
product (DH-09). Never diff a badly aligned scan: `register()` returns ok=False with the reason instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Registration:
    T: np.ndarray            # 4x4, maps current -> baseline frame
    fitness: float           # share of current points with a baseline match within max_corr_m
    rmse: float              # RMS point-to-plane residual of the matched points, metres
    iterations: int
    ok: bool
    reason: str = ""
    structure_fitness: float = float("nan")   # the same share, counted only on points OFF the dominant plane

    def as_dict(self) -> dict:
        return {"fitness": round(self.fitness, 4), "structure_fitness": round(self.structure_fitness, 4),
                "rmse_m": round(self.rmse, 5), "iterations": self.iterations,
                "ok": self.ok, "reason": self.reason, "T": [[round(float(v), 6) for v in r] for r in self.T]}


def estimate_normals(xyz: np.ndarray, k: int = 16, tree: cKDTree | None = None) -> np.ndarray:
    """Unit normals from the smallest-eigenvalue direction of each point's k-neighbourhood covariance."""
    tree = tree or cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k, len(xyz)))
    nb = xyz[idx] - xyz[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def _small_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix for a small rotation vector (exact Rodrigues, so repeated updates stay orthonormal)."""
    v = np.array([rx, ry, rz])
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray | None = None, *,
                       init: np.ndarray | None = None, max_corr_m: float = 0.05, max_iter: int = 200,
                       tol: float = 1e-4) -> tuple[np.ndarray, float, float, int]:
    """Point-to-plane ICP of src onto dst. Returns (T, fitness, rmse, iterations).
    Stops when an update is below `tol` (metres / radians, combined): 1e-4 = 0.1 mm / 0.006 deg. 1e-6 sat below the
    noise floor and kept a fixed rig iterating 83 times for nothing (4.7 s -> 0.6 s, DH-17)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    tree = cKDTree(dst)
    normals = estimate_normals(dst, tree=tree) if dst_normals is None else np.asarray(dst_normals, dtype=np.float64)
    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).copy()
    it = 0
    for it in range(1, max_iter + 1):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p, distance_upper_bound=max_corr_m)
        m = np.isfinite(d)
        if m.sum() < 6:
            break
        p, q, n = p[m], dst[j[m]], normals[j[m]]
        A = np.hstack([np.cross(p, n), n])
        b = np.einsum("ij,ij->i", n, q - p)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dT = np.eye(4)
        dT[:3, :3] = _small_rotation(*x[:3])
        dT[:3, 3] = x[3:]
        T = dT @ T
        if np.linalg.norm(x) < tol:
            break
    p = src @ T[:3, :3].T + T[:3, 3]
    d, j = tree.query(p, distance_upper_bound=max_corr_m)
    m = np.isfinite(d)
    fitness = float(m.mean()) if len(m) else 0.0
    rmse = float(np.sqrt(np.mean(np.einsum("ij,ij->i", normals[j[m]], dst[j[m]] - p[m]) ** 2))) if m.any() else float("inf")
    return T, fitness, rmse, it


def structure_fitness(src_aligned: np.ndarray, dst: np.ndarray, max_corr_m: float, k: int = 16,
                      off_plane_cos: float = 0.7) -> float:
    """Share of src points OFF the dominant plane (|normal . dominant normal| < off_plane_cos) with a dst match.

    Overall fitness cannot see a slide along a big plane (the table matches wherever it is slid to: a 25 cm offset
    still scored 0.82 on the synthetic diorama). Points on box sides / walls only match when the alignment is right.
    The dominant direction is the principal axis of the normals' scatter, so nothing assumes which way is up."""
    n = estimate_normals(src_aligned, k=k)
    w, v = np.linalg.eigh(n.T @ n)
    dom, e1, e2 = v[:, -1], v[:, 0], v[:, 1]
    off = np.abs(n @ dom) < off_plane_cos
    if off.sum() < 30:
        return float("nan")                       # no structure to judge by: reported, not used to reject
    d, _ = cKDTree(dst).query(src_aligned[off], distance_upper_bound=max_corr_m)
    matched = np.isfinite(d)
    # Per face DIRECTION, worst one wins: a slide along long faces still matches them (rack sides slid 17 cm scored
    # 0.72 overall); only faces across the slide fail. Bin the in-plane normal angle (mod 180 deg) into 4 bins.
    ang = np.degrees(np.arctan2(n[off] @ e2, n[off] @ e1)) % 180.0
    bins = (ang // 45).astype(int)
    shares = [matched[bins == b].mean() for b in range(4) if (bins == b).sum() >= 30]
    return float(min(shares)) if shares else float(matched.mean())


def register(current_xyz: np.ndarray, baseline_xyz: np.ndarray, cfg: dict, init: np.ndarray | None = None) -> Registration:
    """ICP with the config's acceptance thresholds:
    cfg['icp'] = {max_rmse_m, min_fitness, min_structure_fitness?, max_corr_m?, max_iter?}."""
    icp = cfg.get("icp", {})
    if len(current_xyz) < 100 or len(baseline_xyz) < 100:
        return Registration(np.eye(4), 0.0, float("inf"), 0, False, "too few points to register")
    max_corr = float(icp.get("max_corr_m", 0.05))
    max_iter = int(icp.get("max_iter", 200))
    src = np.asarray(current_xyz, dtype=np.float64)
    # coarse pass on a small subsample with a wider radius does the long slides cheaply; the fine pass on everything
    # finishes the last millimetres (25 cm knock: 6.6 s -> see DH-17)
    n_coarse = int(icp.get("coarse_points", 4000))
    it_coarse = 0
    if len(src) > 2 * n_coarse:
        sub = src[np.random.default_rng(0).choice(len(src), n_coarse, replace=False)]
        init, _, _, it_coarse = icp_point_to_plane(sub, baseline_xyz, init=init,
                                                   max_corr_m=float(icp.get("coarse_corr_m", 2 * max_corr)), max_iter=max_iter)
    T, fit, rmse, it = icp_point_to_plane(src, baseline_xyz, init=init, max_corr_m=max_corr, max_iter=max_iter)
    it += it_coarse
    sfit = structure_fitness(apply(T, current_xyz), np.asarray(baseline_xyz), max_corr / 2)
    min_fit, max_rmse = float(icp.get("min_fitness", 0.8)), float(icp.get("max_rmse_m", 0.01))
    min_sfit = float(icp.get("min_structure_fitness", 0.9))
    reasons = []
    if fit < min_fit:
        reasons.append(f"fitness {fit:.3f} < {min_fit}")
    if np.isfinite(sfit) and sfit < min_sfit:
        reasons.append(f"structure fitness {sfit:.3f} < {min_sfit}")
    if rmse > max_rmse:
        reasons.append(f"rmse {rmse * 1000:.1f} mm > {max_rmse * 1000:.1f} mm")
    return Registration(T, fit, rmse, it, not reasons, "; ".join(reasons), sfit)


def apply(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
