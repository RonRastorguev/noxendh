"""Generic looming obstacle detection for the Scout's single forward camera (docs/SCOUT_NAV.md §5;
decisions docs/decisions_scout_nav/looming.md, DN-20..DN-37).

Per frame (~10 Hz). The caller passes the FC odometry INTERPOLATED AT THE IMAGE STAMP (not the latest sample).
1. Track Shi-Tomasi corners with pyramidal KLT + forward-backward check (`tracker.FeatureTracker`), new corners
   every frame wherever no track is near.
2. Window k = 3 frames (2 for young tracks). Old positions are DE-ROTATED with the relative camera rotation from the
   FC's odometry (ray rotation = the K R K^-1 homography), so yaw / pitch-on-acceleration cannot loom (DN-21).
   What is left is translation: about the FOE e (motion direction, odometry) a static point contracts radially,
   x_old - e = (x - e) Z / (Z + tau), tau = forward displacement over the window.
3. Static-point checks (DN-24): a de-rotated track off the line through e (slide, T-junction) is dropped, and so is a
   feature without gradient energy both along and across that line (a slide ALONG it would read as wrong depth).
4. Scale change from NEIGHBOUR PAIRS along the TANGENT to the ray from e (DN-22): t.(x_j - x_i)_now /
   t.(x_j - x_i)_old - 1 = tau / Z_j exactly for any surface orientation; residual rotation cancels in the pair.
   The pair measures the NEIGHBOUR j (the anchor's depth cancels), so a feature's own value is the weighted median
   over the pairs in which it is the neighbour (DN-35; a near object beside far background keeps its own depth)
   -> depth Z = tau / w.
5. EMERGENCY (image + FC odometry only; never the map or the PnP pose, DN-25): features whose 3D point (ray x Z,
   body frame) lies in the hull corridor along the odom velocity (an ellipse: the hull is a flat disc); a CLUSTER (>= min_emergency_features,
   image-linked, consistent depth) with median HULL-CONTACT time < ttc_emergency_s -> emergency.
6. MAP-ANCHORED (with T_map_body, DN-26/27): the map is rendered at the camera pose (min-filtered over the
   pose-error radius: the NEAREST mapped surface nearby); every pair's measured expansion minus the expansion the
   map predicts for the same pair (known odom ego-motion; its scale measured on the mapped scene, DN-36) ->
   per-feature excess (weighted median, per-track low-pass). Old positions here come from a DIRECT k-frame KLT on the rotation-warped old image (DN-23).
   Features expanding significantly faster than the map explains (not on a mapped depth edge, TTC <
   map_horizon_s, for >= map_confirm_frames frames) are clustered, placed at their own depth, and boxed in map
   (DN-29/30: floor to top; top bounded by background rays seen above it, else out of view -> ceiling).
7. No closing motion (optical-axis speed < min_closing_speed_mps) -> nothing new (DN-28). An FC local-frame reset
   (reset_counter change, or an odom jump the velocity cannot explain) drops the motion history (DN-34).

All numbers are PROVISIONAL (synthetic data only). Python 3.9, numpy/scipy/OpenCV, no ROS.
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from localization.depth_render import render_depth
from localization.geometry import R_FLU_OPTICAL, inv_T, make_T
from navigation.types import LoomingOutput, Obstacle

from .tracker import FeatureTracker

DEFAULTS = {
    # --- shared values (read from their owners by avoidance.config.load_config; these mirror them for tests) ---
    "ttc_emergency_s": 1.5,          # flight_demo.yaml avoid.looming_ttc_s
    "hull_radius_m": 0.20,           # scout_hw.yaml airframe.hull_radius_m (PROVISIONAL placeholder)
    "hull_half_height_m": 0.06,      # scout_hw.yaml airframe.hull_half_height_m (PROVISIONAL placeholder)
    "top_clear_m": 0.70,             # hull_radius_m + flight_demo.yaml nav.dynamic_inflate_extra_m
    # --- tracking ---
    "max_features": 400,
    "quality_level": 0.01,
    "min_distance_px": 10,
    "klt_win_px": 21,
    "klt_levels": 3,
    "fb_max_px": 0.7,
    "border_px": 6,
    # --- scale change ---
    "window_frames": 3,              # longest window (frames); a feature uses the longest it has
    "min_window_frames": 2,          # youngest usable track (a new obstacle is measurable 0.2 s after it appears)
    "direct_check_px": 1.0,          # direct de-rotated KLT vs chained track must agree within this; 0 = off
    "direct_max_w": 0.06,            # above this expansion the chained track is used (patch scale change)
    "neighbours_k": 16,              # 10 -> 16 and 80 -> 120 px with own-depth pairs (DN-35): an anchor's depth cancels,
    "max_pair_px": 120.0,            # so longer pairs add precision without mixing depths
    "min_proj_px": 6.0,              # a pair's projected length must be at least this (noise / length)
    "min_pairs": 3,
    "own_depth": True,               # a feature's depth from the pairs that measure IT (DN-35); False = its neighbours'
    "map_own": True,                 # ... and the map path's excess / expansions the same way (DN-35)
    "foe_radius_px": 25.0,           # inside this radius of the FOE the pair direction replaces the tangent
    "min_closing_speed_mps": 0.10,   # below: no looming, nothing new
    "min_window_motion_m": 0.02,
    "odom_jump_m": 0.08,             # odom displacement minus velocity x dt above this = a local-frame reset
    "max_frame_gap_s": 0.5,
    "reset_settle_frames": 2,        # frames after a reset whose odometry is not trusted as a window start          # a longer gap between frames also restarts the motion history     # forward displacement over the window needed to measure anything
    "w_min": 0.006,                  # scale changes below this are noise (no depth)
    "aperture_min_frac": 0.1,        # gradient energy along AND across the epipolar line, each >= this x total (DN-35)
    "aperture_win_px": 9,
    "epi_tol_px": 1.0,               # off-epipolar-line tolerance of a de-rotated track (static-point check)
    "foe_err_px": 25.0,              # FOE uncertainty the check allows for (odom velocity direction ~4 deg)
    "image_foe": True,               # motion direction from the image's FOE, magnitude from the odometry (DN-37)
    "foe_min_flow_px": 0.5,          # ... tracks with at least this much de-rotated flow
    "foe_min_tracks": 30,            # ... at least this many (else the odometry's direction)
    "foe_huber_px": 1.0,             # ... robust (Huber) scale of the line residuals
    # --- emergency (map-free) ---
    "corridor_margin_m": 0.15,       # corridor radius = hull_radius_m + this
    "min_emergency_features": 5,
    "min_emergency_inside": 3,       # ... of which at least this many inside the flat-disc corridor ellipse
    "emergency_link_px": 60.0,
    "cluster_depth_tol": 0.35,       # linked features' depths agree within this fraction
    "ttc_report_cap_s": 10.0,
    "raw_explained_frac": 0.5,       # no looming_raw box when this share of the emergency cluster is map-consistent
    # --- map-anchored ---
    "map_render_scale": 0.25,        # expected depth rendered at this fraction of the image resolution
    "map_render_spacing_m": 0.06,    # map voxel-downsampled to this for rendering
    "map_max_range_m": 12.0,
    "map_edge_ratio": 1.25,          # max / min map depth in that radius above this = a mapped depth edge
    "map_pair_depth_ratio": 1.5,     # map-path pairs only between features whose map depths are within this
    "map_pose_tol_px": 2,            # min-filter radius on the rendered depth (render px; 2 x 4 = 8 px ~ 1.4 deg)
    "map_neighbours_k": 16,
    "map_max_pair_px": 120.0,
    "map_min_proj_px": 12.0,
    "map_min_pairs": 4,
    "map_filter_alpha": 0.5,         # per-track low-pass of the excess expansion (1 = no filter)
    "ego_scale_max": 0.3,            # odometry scale from the mapped scene, clamped to 1 +- this (DN-36); 0 = off
    "ego_scale_min_features": 40,    # ... from at least this many map-consistent features
    "ego_scale_min_w": 0.01,         # ... whose predicted expansion is at least this (far features are noise)
    "map_rel_margin": 0.35,          # flagged when w_meas - w_pred > max(map_abs_margin, map_rel_margin * w_pred)
    "map_abs_margin": 0.012,
    "map_horizon_s": 10.0,           # flag only what would be reached within this (at the current closing speed)
    "map_min_cluster_features": 6,
    "map_link_px": 50.0,
    "map_min_tau_m": 0.07,           # map path needs >= this forward motion per window (~0.25 m/s at 10 Hz)
    "map_border_px": 24,             # features this close to the image border are never flagged (one-sided pairs)
    "map_place_quantile": 0.3,       # cluster depth = this quantile of its members' own depths (near side)
    "map_merge_gap_m": None,         # clusters closer than this (3D) are one obstacle; None = hull diameter
    "map_place_tol_m": 0.3,          # consecutive placements of one obstacle must agree within max(this,
    "map_place_tol_rel": 0.15,       # ... this x range) before it is emitted
    "map_confirm_frames": 3,         # median flag streak of a cluster's tracks (persistence before emitting)
    "assumed_depth_m": 0.5,          # unseen extent of an obstacle along the view
    "edge_pad_m": 0.05,
    "top_min_features": 3,           # background rays above the cluster needed to bound its top (k-th lowest)
    "top_bg_depth_factor": 1.5,     # ... and each must measure at least this x the cluster depth (behind it)
    "top_bg_map_tol": 0.15,          # ... or its own depth within this fraction of its map depth
    "top_near_depth_factor": 1.2,    # features above within this x the cluster depth are more obstacle ...
    "top_near_min_features": 2,      # ... and this many of them make its top unknown (-> ceiling)
    "top_edge_px": 100,              # cluster top closer than this to the image top: treat its top as out of view
    "top_unseen_pad_m": 0.5,         # top in view, nothing near above it, but not bounded: feature top + this
    "debug_arrays": False,           # put per-feature arrays in LoomingOutput.debug (diagnostics only)
}


def _voxel_down(pts, s):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if s <= 0 or len(pts) == 0:
        return pts.astype(np.float32)
    key = np.floor(pts / s).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(first)].astype(np.float32)


def _wmedian(r, good, wt, min_n):
    """Weighted median of r over `good` per row (weights wt: longer pairs count more); NaN where < min_n pairs."""
    n = good.sum(axis=1)
    w = np.where(good, wt, 0.0)
    order = np.argsort(np.where(good, r, np.inf), axis=1)
    rs = np.take_along_axis(r, order, axis=1)
    cw = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
    half = 0.5 * cw[:, -1:]
    j = np.minimum((cw < half).sum(axis=1), r.shape[1] - 1)
    med = rs[np.arange(len(r)), j]
    med[n < min_n] = np.nan
    return med


def _wmedian_own(r, good, wt, nb, n, min_n):
    """Per-feature OWN expansion (DN-35): pair (i, j) projected on the tangent at the anchor i measures tau / Z_j,
    the NEIGHBOUR's depth (the anchor's cancels). So feature j's own value is the weighted median over the anchors
    that have j as a neighbour, whatever their depths: a feature on a near object next to far background is not
    pulled toward the background. -> (n,) NaN where fewer than min_n pairs."""
    out = np.full(n, np.nan)
    tgt, v, w = nb[good], r[good], wt[good]
    if not len(v):
        return out
    order = np.lexsort((v, tgt))
    tgt, v, w = tgt[order], v[order], w[order]
    cnt = np.bincount(tgt, minlength=n)
    tot = np.bincount(tgt, weights=w, minlength=n)
    cw = np.cumsum(w)
    before = np.concatenate([[0.0], cw])[np.concatenate([[0], np.cumsum(cnt)[:-1]])]
    pos = np.nonzero(cw - before[tgt] >= 0.5 * tot[tgt])[0]
    g, first = np.unique(tgt[pos], return_index=True)
    out[g] = v[pos[first]]
    out[cnt < min_n] = np.nan
    return out


def _clusters(p, z, link_px, depth_tol):
    """Connected components: image distance < link_px AND depths within depth_tol (relative). -> labels."""
    n = len(p)
    if n == 0:
        return np.zeros(0, np.int64)
    pairs = cKDTree(p).query_pairs(link_px, output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        ok = np.abs(z[a] - z[b]) <= depth_tol * np.minimum(z[a], z[b])
        a, b = a[ok], b[ok]
    else:
        a = b = np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


class LoomingDetector:
    def __init__(self, cam, cfg: dict | None = None, map_points=None, map_spacing_m: float = 0.03,
                 T_body_cam=None):
        """`cam`: CameraModel of the stream. `map_points`: (N, 3) prior map in MAP (None = emergency path only).
        `T_body_cam`: camera mount (body FLU <- optical); None = optical axis along body x, zero offset."""
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        c = self.cfg
        self.cam = cam
        self.f = 0.5 * (cam.fx + cam.fy)
        self.T_body_cam = np.asarray(T_body_cam, dtype=np.float64) if T_body_cam is not None \
            else make_T(R_FLU_OPTICAL, (0.0, 0.0, 0.0))
        self.W = int(c["window_frames"])
        self.tracker = FeatureTracker(self.W + 1, c["max_features"], c["quality_level"], c["min_distance_px"],
                                      7, c["klt_win_px"], c["klt_levels"], c["fb_max_px"], c["border_px"])
        self.tracker.add_attr("streak", 0, np.int32)
        self.tracker.add_attr("obs_id", -1, np.int64)
        self.tracker.add_attr("ex_f", np.nan, np.float64)
        self.tracker.add_attr("wp_f", np.nan, np.float64)
        self.frames = deque(maxlen=self.W + 1)          # (stamp, T_odom_cam), newest first
        self.map_pts = None
        self.floor_z = self.ceiling_z = None
        if map_points is not None and len(map_points):
            mp = np.asarray(map_points, dtype=np.float64).reshape(-1, 3)
            self.render_spacing = max(float(map_spacing_m), float(c["map_render_spacing_m"]))
            self.map_pts = _voxel_down(mp, float(c["map_render_spacing_m"]))
            self.cam_small = cam.scaled(float(c["map_render_scale"]))
            half = np.radians(cam.fov_deg()["diagonal"] / 2.0 + 10.0)
            self._cos_cone = np.float32(np.cos(min(half, np.radians(89.0))))
            self.floor_z = float(np.percentile(mp[:, 2], 0.5))
            self.ceiling_z = float(np.percentile(mp[:, 2], 99.5))
        self._next_obs = 0
        self._first_seen = {}
        self._reset_counter = None
        self._settle = 0
        self._placed = {}                            # obstacle id -> (face xy in map, stamp) of its last placement

    def reset(self):
        self.tracker.reset()
        self.tracker.add_attr("streak", 0, np.int32)
        self.tracker.add_attr("obs_id", -1, np.int64)
        self.tracker.add_attr("ex_f", np.nan, np.float64)
        self.tracker.add_attr("wp_f", np.nan, np.float64)
        self.frames.clear()

    # ----------------------------------------------------------------------------------------------------------
    def _pairs(self, p_now, p_old_sets, e, nb, nbvalid, min_proj):
        """Per-pair expansion r_ij = t.d_now / t.d_old - 1 (= tau / Z_j for a static point j) for each set of
        de-rotated old positions; t = the tangent to the ray from the FOE e at feature i (the pair's own direction
        near e). Pairs are kept only where every set has a usable projection, so sets compare like for like.
        -> (list of r (n, K), good (n, K), weight |t.d_now| (n, K))."""
        c = self.cfg
        rel = p_now - e
        rn = np.linalg.norm(rel, axis=1)
        near = rn < c["foe_radius_px"]
        tang = np.stack([-rel[:, 1], rel[:, 0]], 1) / np.maximum(rn, 1e-9)[:, None]
        dnew = p_now[nb] - p_now[:, None, :]
        dn = np.linalg.norm(dnew, axis=2, keepdims=True)
        own = dnew / np.maximum(dn, 1e-9)
        dirv = np.where(near[:, None, None], own, tang[:, None, :])
        a_new = np.sum(dirv * dnew, axis=2)
        good = nbvalid & (np.abs(a_new) >= min_proj)
        a_olds = []
        for p_old in p_old_sets:
            dold = p_old[nb] - p_old[:, None, :]
            a_old = np.sum(dirv * dold, axis=2)
            good &= np.isfinite(a_old) & (a_old * a_new > 0)
            a_olds.append(a_old)
        rs = [np.where(good, a_new / np.where(good, a_old, 1.0) - 1.0, 0.0) for a_old in a_olds]
        return rs, good, np.abs(a_new)

    def _structure_tensor(self, gray, uv):
        """Per-feature 2x2 structure tensor (Sxx, Syy, Sxy) over a klt-window-sized box."""
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        w = int(self.cfg["aperture_win_px"]) // 2
        h_, w_ = g.shape
        off = np.arange(-w, w + 1)
        uu = np.clip(np.rint(uv[:, 0]).astype(int)[:, None, None] + off[None, None, :], 0, w_ - 1)
        vv = np.clip(np.rint(uv[:, 1]).astype(int)[:, None, None] + off[None, :, None], 0, h_ - 1)
        px, py = gx[vv, uu], gy[vv, uu]                     # (n, 2w+1, 2w+1) patches at the features only
        out = [(px * px).sum(axis=(1, 2)), (py * py).sum(axis=(1, 2)), (px * py).sum(axis=(1, 2))]
        return np.stack(out, 1)

    def _two_way_texture(self, S, rel):
        """Aperture check (DN-24): the feature must have gradient energy BOTH along the epipolar (radial) direction
        and across it. A stroke parallel to the epipolar line lets KLT slide along it, which the static-point check
        cannot see (the slide IS motion along the line) and which reads as a wrong depth."""
        rn = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
        ux, uy = rel[:, 0] / rn, rel[:, 1] / rn
        sxx, syy, sxy = S[:, 0], S[:, 1], S[:, 2]
        e_r = ux * ux * sxx + 2 * ux * uy * sxy + uy * uy * syy
        e_t = uy * uy * sxx - 2 * ux * uy * sxy + ux * ux * syy
        tr = np.maximum(sxx + syy, 1e-9)
        f = self.cfg["aperture_min_frac"]
        return (e_r >= f * tr) & (e_t >= f * tr)

    def _epipolar_ok(self, p_now, p_old, e):
        """Static-point check (DN-24): after de-rotation a static point moves exactly along the line through the
        FOE e. A track whose old position is off that line by more than epi_tol_px (+ what an FOE error of
        foe_err_px explains) slid along an edge (aperture problem) or is a T-junction: it is used for nothing."""
        c = self.cfg
        rel = p_now - e
        rn = np.maximum(np.linalg.norm(rel, axis=1), c["foe_radius_px"])
        o = p_old - e
        perp = np.abs(o[:, 0] * rel[:, 1] - o[:, 1] * rel[:, 0]) / rn
        disp = np.linalg.norm(p_now - p_old, axis=1)
        tol = c["epi_tol_px"] + c["foe_err_px"] * disp / rn
        return np.isfinite(perp) & (perp <= tol)

    def _image_foe(self, ps, flow, e0):
        """Focus of expansion of the de-rotated flow (DN-37): the point that the lines through every track along its
        flow pass closest to (weighted least squares, Huber-reweighted). -> ideal px (2,), or None when the flow is
        too small / too few tracks / not an expansion (then the odometry's FOE is used)."""
        c = self.cfg
        L = np.linalg.norm(flow, axis=1)
        ok = np.isfinite(L) & (L >= c["foe_min_flow_px"])
        if ok.sum() < c["foe_min_tracks"]:
            return None
        p, fl, L = ps[ok], flow[ok], L[ok]
        nrm = np.stack([-fl[:, 1], fl[:, 0]], 1) / L[:, None]
        w0 = np.minimum(L, 20.0)
        w = w0.copy()
        e = np.asarray(e0, dtype=np.float64).copy()
        for _ in range(5):
            A = (nrm * w[:, None]).T @ nrm
            if np.linalg.cond(A) > 1e6:
                return None
            e = np.linalg.solve(A, (nrm * w[:, None]).T @ np.sum(nrm * p, axis=1))
            res = np.abs(np.sum(nrm * (p - e), axis=1))
            w = w0 * np.minimum(1.0, c["foe_huber_px"] / np.maximum(res, 1e-9))
        inl = res <= 3.0 * c["foe_huber_px"]
        if inl.sum() < c["foe_min_tracks"] or np.mean(np.sum(fl[inl] * (p[inl] - e), axis=1) > 0) < 0.8:
            return None
        return e

    def _undist_maps(self, R):
        cam = self.cam
        return cv2.initUndistortRectifyMap(cam.K, cam.dist, R, cam.K, (cam.width, cam.height), cv2.CV_32FC1)

    def _direct_derotated(self, k, R, idx, p_chain, gray_u, n_now):
        """De-rotated old positions measured DIRECTLY (DN-23): image k is warped into the CURRENT camera
        orientation with the FC's relative rotation (K R K^-1; for a lens with distortion an undistort+rotate
        remap), then ONE KLT step from the current image finds each feature there. Rotation no longer deforms the
        patches (a 102 deg lens stretches the periphery by ~2 x theta under a yaw) and k chained steps of drift do
        not accumulate. ok = converged, inside, and within direct_check_px of the chained, de-rotated track.
        -> (ideal px (n, 2), ok (n,))."""
        cam, c = self.cam, self.cfg
        g_old = self.tracker.grays[k]
        if cam.has_distortion:
            warped = cv2.remap(g_old, *self._undist_maps(R), cv2.INTER_LINEAR)
        else:
            H = cam.K @ R @ np.linalg.inv(cam.K)
            warped = cv2.warpPerspective(g_old, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR)
        fxy = np.array([cam.fx, cam.fy])
        cxy = np.array([cam.cx, cam.cy])
        p0 = (n_now[idx] * fxy + cxy).astype(np.float32).reshape(-1, 1, 2)
        init = np.nan_to_num(p_chain / self.f * fxy + cxy, nan=-1.0).astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_u, warped, p0, init.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                             **self.tracker.lk)
        p1 = p1.reshape(-1, 2).astype(np.float64)
        b = 2.0 + 0.5 * self.tracker.lk["winSize"][0]
        inside = (p1[:, 0] >= b) & (p1[:, 0] <= cam.width - 1 - b) & (p1[:, 1] >= b) & (p1[:, 1] <= cam.height - 1 - b)
        p_dir = (p1 - cxy) / fxy * self.f
        dev = np.linalg.norm(p_dir - p_chain, axis=1)
        ok = (st.reshape(-1) == 1) & inside & np.isfinite(dev) & (dev < c["direct_check_px"])
        return p_dir, ok

    def _new_obs_id(self, track_ids_obs, stamp):
        prev = track_ids_obs[track_ids_obs >= 0]
        if len(prev):
            vals, cnt = np.unique(prev, return_counts=True)
            oid = int(vals[np.argmax(cnt)])
        else:
            oid = self._next_obs
            self._next_obs += 1
        self._first_seen.setdefault(oid, stamp)
        return oid

    # ----------------------------------------------------------------------------------------------------------
    def _motion_reset(self):
        """Forget every odometry delta (the FC reset its local frame): tracks keep only their current position,
        so nothing is measured across the jump; the per-track filters restart."""
        tr = self.tracker
        if len(self.frames) > 1:
            newest = self.frames[0]
            self.frames.clear()
            self.frames.appendleft(newest)
        tr.hist[:, 1:] = np.nan
        if len(tr.grays) > 1:
            g = tr.grays[0]
            tr.grays.clear()
            tr.grays.appendleft(g)
        tr.set_attr("streak", np.zeros(len(tr.ids)))
        tr.set_attr("ex_f", np.full(len(tr.ids), np.nan))
        tr.set_attr("wp_f", np.full(len(tr.ids), np.nan))

    def process(self, image, stamp, T_odom_body, v_odom, T_map_body=None, reset_counter=None) -> LoomingOutput:
        """`T_odom_body`, `v_odom`: FC odometry interpolated AT THE IMAGE STAMP. `T_map_body`: map pose at the same
        stamp (None = emergency path only). `reset_counter`: the FC's local-frame reset counter (OdomSample);
        a change drops the motion history (DN-34). An odometry jump the velocity cannot explain does the same."""
        t0 = time.perf_counter()
        c = self.cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        T_odom_body = np.asarray(T_odom_body, dtype=np.float64)
        T_odom_cam = T_odom_body @ self.T_body_cam
        v_odom = np.asarray(v_odom, dtype=np.float64).reshape(3)
        tstats = self.tracker.step(gray)
        self.frames.appendleft((float(stamp), T_odom_cam, T_odom_body[:3, 3].copy(), v_odom.copy()))
        reset_why = None
        if reset_counter is not None and self._reset_counter is not None and reset_counter != self._reset_counter:
            reset_why = "reset_counter"
        if reset_counter is not None:
            self._reset_counter = reset_counter
        if reset_why is None and len(self.frames) > 1:
            t1, _, p1, v1 = self.frames[1]
            dt = float(stamp) - t1
            jump = np.linalg.norm((T_odom_body[:3, 3] - p1) - 0.5 * (v_odom + v1) * dt)
            if dt <= 0 or dt > c["max_frame_gap_s"] or jump > c["odom_jump_m"]:
                reset_why = "odom_jump" if dt > 0 and dt <= c["max_frame_gap_s"] else "frame_gap"
        if reset_why is None and self._settle > 0:
            reset_why = "settling"                   # odometry right at a reset may still be mis-timed (DN-34)
            self._settle -= 1
        elif reset_why is not None:
            self._settle = int(c["reset_settle_frames"])
        if reset_why is not None:
            self._motion_reset()
        tr = self.tracker
        out = LoomingOutput(float(stamp), False, float("inf"), [], n_tracks=len(tr.ids))
        dbg = out.debug
        dbg["tracker"] = tstats
        if reset_why is not None:
            dbg["motion_reset"] = reset_why
        speed = float(np.linalg.norm(v_odom))
        closing = float((T_odom_cam[:3, :3].T @ v_odom)[2])
        dbg["closing_mps"] = closing

        def done(reason):
            out.reason = reason
            dbg["ms"] = 1000.0 * (time.perf_counter() - t0)
            return out

        if closing < c["min_closing_speed_mps"]:
            tr.set_attr("streak", np.zeros(len(tr.ids)))
            tr.set_attr("ex_f", np.full(len(tr.ids), np.nan))
            tr.set_attr("wp_f", np.full(len(tr.ids), np.nan))
            return done("no_closing_motion")
        n_hist = len(self.frames)
        kmax = min(self.W, n_hist - 1)
        if kmax < c["min_window_frames"] or len(tr.ids) < 10:
            return done("warming_up")

        M = len(tr.ids)
        uv_now = tr.hist[:, 0].astype(np.float64)
        n_now = self.cam.normalized(uv_now)
        p_now = n_now * self.f
        T_cam_odom = inv_T(T_odom_cam)

        # map depth at the features (expected depth of the PRIOR map at the camera pose)
        use_map = T_map_body is not None and self.map_pts is not None
        if use_map:
            T_map_cam = np.asarray(T_map_body, dtype=np.float64) @ self.T_body_cam
            tm = time.perf_counter()
            # cull to the view cone first (render_depth transforms every point it is given)
            rel = self.map_pts - T_map_cam[:3, 3].astype(np.float32)
            fwd = T_map_cam[:3, 2].astype(np.float32)
            along = rel @ fwd
            cone = along > self._cos_cone * np.sqrt(np.einsum("ij,ij->i", rel, rel))
            dimg = render_depth(self.map_pts[cone], T_map_cam, self.cam_small, self.render_spacing,
                                max_z=c["map_max_range_m"])
            s = float(c["map_render_scale"])
            # CONSERVATIVE expected depth (DN-27): the NEAREST mapped surface within the pose-error radius, so a
            # few cm / ~1 deg of map-pose error at a mapped depth edge predicts MORE expansion, never less
            # A feature whose window spans a mapped depth edge (max / min > map_edge_ratio) may be a T-junction
            # (a corner made by a near edge over a far texture, which does not move like any point): it serves as
            # a neighbour but is never flagged itself.
            r_px = max(int(c["map_pose_tol_px"]), 1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
            fin = np.isfinite(dimg)
            dmin = cv2.erode(np.where(fin, dimg, np.float32(1e6)).astype(np.float32), ker)
            dmax = cv2.dilate(np.where(fin, dimg, np.float32(0)).astype(np.float32), ker)
            us = np.clip(np.rint((uv_now[:, 0] + 0.5) * s - 0.5).astype(int), 0, dimg.shape[1] - 1)
            vs = np.clip(np.rint((uv_now[:, 1] + 0.5) * s - 0.5).astype(int), 0, dimg.shape[0] - 1)
            z_map = dmin[vs, us].astype(np.float64)
            z_map[z_map >= 1e6] = np.nan
            map_edge = ~(dmax[vs, us] <= c["map_edge_ratio"] * z_map)
            dbg["render_ms"] = 1000.0 * (time.perf_counter() - tm)
        w_meas = np.full(M, np.nan)       # expansion, emergency neighbourhood (small: near surfaces only)
        wm_meas = np.full(M, np.nan)      # expansion, map neighbourhood (larger: far range needs long pairs)
        ex_now = np.full(M, np.nan)       # per-pair (measured - map-predicted) expansion, weighted median
        wm_pred = np.full(M, np.nan)
        tau_of = np.full(M, np.nan)
        k_of = np.zeros(M, np.int64)
        epi_of = np.zeros(M, bool)
        Ke, Km = int(c["neighbours_k"]), int(c["map_neighbours_k"]) if use_map else int(c["neighbours_k"])
        Rq = max(c["max_pair_px"], c["map_max_pair_px"]) if use_map else c["max_pair_px"]
        gray_u = gray
        tensor = self._structure_tensor(gray, uv_now) if c["aperture_min_frac"] > 0 else None
        if use_map and c["direct_check_px"] > 0 and self.cam.has_distortion:
            gray_u = cv2.remap(gray, *self._undist_maps(np.eye(3)), cv2.INTER_LINEAR)
        u_img = None                                     # body motion direction from the image FOE (longest window)
        for k in range(kmax, int(c["min_window_frames"]) - 1, -1):
            T_now_old = T_cam_odom @ self.frames[k][1]
            R = T_now_old[:3, :3]
            m = -T_now_old[:3, 3]                       # camera motion old -> now, in the current camera frame
            if m[2] < c["min_window_motion_m"]:
                continue
            idx = np.nonzero(np.isfinite(tr.hist[:, k, 0]))[0]
            if len(idx) < Ke:
                continue
            n_old = self.cam.normalized(tr.hist[idx, k].astype(np.float64))
            d = np.column_stack([n_old, np.ones(len(idx))]) @ R.T
            p_old = d[:, :2] / d[:, 2:3] * self.f              # chained track, de-rotated (ideal px)
            ps = p_now[idx]
            # motion DIRECTION from the image (DN-37): the de-rotated flow's focus of expansion. The odometry keeps
            # the magnitude (scale-free for the TTC: depth and speed carry the same scale error), taken from its
            # HORIZONTAL displacement: the FC's z is height above the surface under the rangefinder (DN-03) and
            # steps at every terrain edge (leaving the pad: +0.3 m of "climb" in 0.3 s), the flow xy does not
            e_img = self._image_foe(ps, ps - p_old, m[:2] / m[2] * self.f) if c["image_foe"] else None
            if e_img is not None:
                t_dir = np.array([e_img[0] / self.f, e_img[1] / self.f, 1.0])
                t_dir /= np.linalg.norm(t_dir)
                R_oc = T_odom_cam[:3, :3]
                t_w, m_w = R_oc @ t_dir, R_oc @ m
                h_t = float(np.hypot(t_w[0], t_w[1]))
                mag = float(np.hypot(m_w[0], m_w[1])) / h_t if h_t > 0.3 else max(float(m_w @ t_w), 0.0)
                m = mag * t_dir
                if u_img is None:
                    R_bc_ = self.T_body_cam[:3, :3]
                    # body displacement in the current body frame: camera displacement minus the lever arm's
                    ub = R_bc_ @ m - (np.eye(3) - R_bc_ @ R @ R_bc_.T) @ self.T_body_cam[:3, 3]
                    if np.linalg.norm(ub) > 1e-6:
                        u_img = ub / np.linalg.norm(ub)
                        dbg["foe_odom_deg"] = float(np.degrees(np.arccos(np.clip(
                            (m / np.linalg.norm(m)) @ (-T_now_old[:3, 3]) / max(np.linalg.norm(T_now_old[:3, 3]),
                                                                              1e-12), -1.0, 1.0))))
            tau = float(m[2])
            if tau < c["min_window_motion_m"]:
                continue
            e = m[:2] / tau * self.f
            dist, nb = cKDTree(ps).query(ps, k=min(max(Ke, Km) + 1, len(idx)), distance_upper_bound=Rq)
            dist, nb = dist[:, 1:], nb[:, 1:]
            nbvalid = np.isfinite(dist)
            nb = np.where(nbvalid, nb, 0)
            tgt = k_of[idx] == 0
            sel = idx[tgt]
            # emergency: CHAINED tracks (frame-to-frame KLT; at short range a k-frame step would see a 10-15 %
            # patch scale change, which a translation-only KLT gets wrong by up to a pixel, DN-23)
            epi = self._epipolar_ok(ps, p_old, e)
            if tensor is not None:
                epi &= self._two_way_texture(tensor[idx], ps - e)
            epi_of[sel] = epi[tgt]
            ne = nbvalid[:, :Ke] & (dist[:, :Ke] <= c["max_pair_px"]) & epi[nb[:, :Ke]] & epi[:, None]
            (r,), good, wt = self._pairs(ps, [p_old], e, nb[:, :Ke], ne, c["min_proj_px"])
            if c["own_depth"]:
                w_ch = _wmedian_own(r, good, wt, nb[:, :Ke], len(idx), int(c["min_pairs"]))
            else:
                w_ch = _wmedian(r, good, wt, int(c["min_pairs"]))
            w_meas[sel] = w_ch[tgt]
            tau_of[sel] = tau
            k_of[sel] = k
            if use_map:
                # map path: far range, small signal -> the DIRECT de-rotated measurement where the expansion is
                # small (no drift, no rotation deformation); chained where it is large; dropped where they disagree
                p_m = p_old
                if c["direct_check_px"] > 0:
                    p_dir, okd = self._direct_derotated(k, R, idx, p_old, gray_u, n_now)
                    big = np.isfinite(w_ch) & (w_ch >= c["direct_max_w"])
                    p_m = np.where(big[:, None], p_old, np.where(okd[:, None], p_dir, np.nan))
                zm = z_map[idx]
                hm = np.isfinite(zm)
                X = np.column_stack([n_now[idx], np.ones(len(idx))]) * np.where(hm, zm, 1.0)[:, None] + m
                p_pred = np.where(hm[:, None], X[:, :2] / X[:, 2:3] * self.f, np.nan)
                hm &= self._epipolar_ok(ps, p_m, e)
                nm = nbvalid[:, :Km] & hm[nb[:, :Km]] & hm[:, None]
                # pairs only within one mapped depth layer: a pair across a mapped occlusion edge mixes a near and a
                # far surface, where tracking errors on either side spread excess to the other (DN-27)
                zi = np.where(hm, zm, 1.0)
                zr = zi[nb[:, :Km]] / zi[:, None]
                nm &= (zr <= c["map_pair_depth_ratio"]) & (zr >= 1.0 / c["map_pair_depth_ratio"])
                (rm, rp), good, wt = self._pairs(ps, [p_m, p_pred], e, nb[:, :Km], nm, c["map_min_proj_px"])
                mp = int(c["map_min_pairs"])
                if c["own_depth"] and c["map_own"]:      # each feature's OWN excess (DN-35): no inheritance from
                    def med(r):                          # an obstacle next to it (DN-26's background-feature case)
                        return _wmedian_own(r, good, wt, nb[:, :Km], len(idx), mp)
                else:
                    def med(r):
                        return _wmedian(r, good, wt, mp)
                wm_meas[sel] = med(rm)[tgt]
                wm_pred[sel] = med(rp)[tgt]
                ex_now[sel] = med(rm - rp)[tgt]
        dbg["window_frames"] = int(k_of.max()) if M else 0

        valid = np.isfinite(w_meas) & (w_meas > c["w_min"]) & epi_of
        Z = np.where(valid, tau_of / np.where(valid, w_meas, 1.0), np.inf)
        X_cam = np.column_stack([n_now, np.ones(M)]) * np.where(valid, Z, 0.0)[:, None]
        dbg["n_valid"] = int(valid.sum())

        # ---------------- emergency: image + FC odometry only ----------------
        R_bc, t_bc = self.T_body_cam[:3, :3], self.T_body_cam[:3, 3]
        X_b = X_cam @ R_bc.T + t_bc
        u = (T_odom_body[:3, :3].T @ v_odom) / max(speed, 1e-9)
        if u_img is not None:
            u = u_img                                    # the image's direction, the odometry's speed (DN-37) ...
            u_w = T_odom_body[:3, :3] @ u
            h_u = float(np.hypot(u_w[0], u_w[1]))
            if h_u > 0.3:                                # ... from its horizontal part (z steps with the terrain)
                speed = float(np.hypot(v_odom[0], v_odom[1])) / h_u
        along = X_b @ u
        w_off = X_b - along[:, None] * u
        # the hull is a flat disc: the corridor cross-section is an ellipse, hull radius + margin across, hull
        # half-height + margin vertically (gravity from the FC attitude), DN-25
        up = T_odom_body[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        up_p = up - (up @ u) * u
        r_hull = float(c["hull_radius_m"])
        a_h = r_hull + c["corridor_margin_m"]
        a_v = float(c["hull_half_height_m"]) + c["corridor_margin_m"]
        if np.linalg.norm(up_p) > 0.3:                      # not a (near-)vertical motion
            up_p /= np.linalg.norm(up_p)
            dv = w_off @ up_p
            dh = np.linalg.norm(w_off - dv[:, None] * up_p, axis=1)
            inside = (dh / a_h) ** 2 + (dv / a_v) ** 2 < 1.0
        else:
            inside = np.linalg.norm(w_off, axis=1) < a_h
        contact = np.maximum(along - r_hull, 0.0) / max(speed, 1e-9)
        # candidates: the round cross-section (hull radius + margin); a cluster counts only if enough of its members
        # are inside the flat-disc ellipse (a surface the hull clears by its height is never "in the way")
        corr = (valid & (along > 0) & (np.linalg.norm(w_off, axis=1) < a_h)
                & (contact < c["ttc_report_cap_s"]))
        dbg["n_corridor"] = int(corr.sum())
        ci = np.nonzero(corr)[0]
        emerg_feats = None
        if len(ci) >= c["min_emergency_features"]:
            lab = _clusters(p_now[ci], Z[ci], c["emergency_link_px"], c["cluster_depth_tol"])
            for L in np.unique(lab):
                mem = ci[lab == L]
                if len(mem) < c["min_emergency_features"] or inside[mem].sum() < c["min_emergency_inside"]:
                    continue
                ttc = float(np.median(contact[mem]))
                if ttc < out.min_ttc_s:
                    out.min_ttc_s = ttc
                    emerg_feats = mem
        out.emergency = bool(out.min_ttc_s < c["ttc_emergency_s"])
        if c.get("debug_arrays"):
            dbg["emerg_idx"] = emerg_feats
            dbg["uv"], dbg["Z"], dbg["contact"] = uv_now, Z, contact
        raw_pending = out.emergency and T_map_body is not None

        # ---------------- map-anchored ----------------
        if use_map:
            out.map_anchored = True
            T_map_cam = np.asarray(T_map_body, dtype=np.float64) @ self.T_body_cam
            # ego-motion scale from the mapped scene (DN-36): the prediction uses the ODOMETRY's displacement, whose
            # flow-scale / velocity-bias error (sim strong drift: 15-20 %) reads as excess expansion on every mapped
            # surface. Most features are on the map, so the median measured / predicted expansion is that scale.
            # Map path only: the emergency keeps the raw odometry (DN-25)
            sc = 1.0
            okr = np.isfinite(wm_meas) & np.isfinite(wm_pred) & (wm_pred > c["ego_scale_min_w"]) & ~map_edge
            if c["ego_scale_max"] > 0 and okr.sum() >= c["ego_scale_min_features"]:
                sc = float(np.clip(np.median(wm_meas[okr] / wm_pred[okr]),
                                   1.0 - c["ego_scale_max"], 1.0 + c["ego_scale_max"]))
            dbg["ego_scale"] = sc
            ex_now = ex_now - (sc - 1.0) * wm_pred
            wm_pred = sc * wm_pred
            both = np.isfinite(ex_now) & np.isfinite(wm_pred)
            # per-track low-pass of the excess and of the prediction (the same surface is seen frame after frame)
            a = float(c["map_filter_alpha"])
            exf, wpf = tr.attr("ex_f"), tr.attr("wp_f")
            fresh = ~np.isfinite(exf)
            exf = np.where(both, np.where(fresh, ex_now, (1 - a) * exf + a * ex_now), np.nan)
            wpf = np.where(both, np.where(fresh, wm_pred, (1 - a) * wpf + a * wm_pred), np.nan)
            tr.set_attr("ex_f", exf)
            tr.set_attr("wp_f", wpf)
            margin = np.maximum(c["map_abs_margin"], c["map_rel_margin"] * np.maximum(wpf, 0.0))
            w_obj = np.where(both, wpf + exf, np.nan)                      # the feature's own expansion
            zf = np.where(both & (w_obj > c["w_min"]), sc * tau_of / np.where(both, np.maximum(w_obj, 1e-9), 1.0),
                          np.inf)
            b = float(c["map_border_px"])
            inner = ((uv_now[:, 0] >= b) & (uv_now[:, 0] <= self.cam.width - 1 - b) & (uv_now[:, 1] >= b)
                     & (uv_now[:, 1] <= self.cam.height - 1 - b))
            enough = np.nan_to_num(tau_of, nan=0.0) >= c["map_min_tau_m"]   # signal well above the noise floor
            flag = both & inner & enough & ~map_edge & (exf > margin) & (zf / max(closing, 1e-9) < c["map_horizon_s"])
            background = both & (np.abs(exf) <= margin)
            streak = np.where(flag, tr.attr("streak") + 1, 0)
            tr.set_attr("streak", streak)
            dbg["n_flagged"] = int(flag.sum())
            if c.get("debug_arrays"):
                dbg["arrays"] = {"uv": uv_now, "w": w_meas, "wm": wm_meas, "wp": wm_pred, "ex": ex_now, "exf": exf,
                                 "wpf": wpf, "tau": tau_of, "k": k_of, "flag": flag, "z": Z, "zmap": z_map,
                                 "zf": zf}
            dbg["n_background"] = int(background.sum())
            fi = np.nonzero(flag)[0]
            if len(fi) >= c["map_min_cluster_features"]:
                lab = _clusters(p_now[fi], zf[fi], c["map_link_px"], c["cluster_depth_tol"])
                obs_ids = tr.attr("obs_id").copy()
                cands = []                                   # (members, depth, front points in map)
                for L in np.unique(lab):
                    mem = fi[lab == L]
                    if len(mem) < c["map_min_cluster_features"] or np.median(streak[mem]) < c["map_confirm_frames"]:
                        continue
                    # place it with the members' OWN small-neighbourhood depth: a background feature next to the
                    # obstacle inherits excess through the wide map neighbourhood but not a near depth (DN-26)
                    # Two depth estimates: the members' small-neighbourhood depth Z places it (the wide, filtered
                    # map-path depth zf is diluted towards the background); they must agree, else the placement is
                    # not trusted this frame. Z also prunes members that only inherited excess from a neighbour.
                    zc = float(np.median(zf[mem]))
                    zs = sc * Z[mem]                         # the small-neighbourhood depth, same scale
                    fin = np.isfinite(zs)
                    if fin.sum() >= c["map_min_cluster_features"]:
                        # a cluster on an UNMAPPED object also holds T-junctions on its silhouette (far background
                        # behind) whose depth is a mix: the object is the near part, so a low quantile places it
                        zs_med = float(np.percentile(zs[fin], 100.0 * c["map_place_quantile"]))
                        if zs_med > (1 + c["cluster_depth_tol"]) * zc or zc > 2.0 * (1 + c["cluster_depth_tol"]) * zs_med:
                            dbg.setdefault("inconsistent_depth", []).append(round(zc, 2))
                            continue
                        zc = zs_med
                        mem = mem[fin & (np.abs(zs - zc) <= c["cluster_depth_tol"] * zc)]
                    if len(mem) < c["map_min_cluster_features"]:
                        continue
                    cands.append([mem, zc])
                # one object seen as several image clusters (sparse texture, close range): merge clusters whose
                # front faces are at a consistent depth and within map_merge_gap_m of each other in 3D
                merged = True
                while merged and len(cands) > 1:
                    merged = False
                    for i in range(len(cands)):
                        for j in range(i + 1, len(cands)):
                            (mi, zi), (mj, zj) = cands[i], cands[j]
                            if abs(zi - zj) > c["cluster_depth_tol"] * min(zi, zj):
                                continue
                            Pi = np.column_stack([n_now[mi], np.ones(len(mi))]) * zi
                            Pj = np.column_stack([n_now[mj], np.ones(len(mj))]) * zj
                            gap = c["map_merge_gap_m"] if c["map_merge_gap_m"] is not None else 2 * c["hull_radius_m"]
                            if cKDTree(Pi).query(Pj, k=1)[0].min() <= gap:
                                m_ = np.concatenate([mi, mj])
                                cands[i] = [m_, float(np.median(np.concatenate([np.full(len(mi), zi),
                                                                                np.full(len(mj), zj)])))]
                                del cands[j]
                                merged = True
                                break
                        if merged:
                            break
                for mem, zc in cands:
                    P = np.column_stack([n_now[mem], np.ones(len(mem))]) * zc
                    P = P @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
                    bg, v_top, n_near = self._background_above(mem, background, uv_now, sc * Z, zc, z_map)
                    ob = self._box(P, T_map_cam, (bg, n_now, zc, v_top, n_near), stamp, "looming_map", mem,
                                   zc / closing)
                    obs_ids[mem] = ob.id
                    # 3D persistence (DN-26): emitted only when this obstacle's face was placed at the same spot
                    # (within max(map_place_tol_m, map_place_tol_rel x range)) in the previous frame as well
                    face = np.asarray(ob.info["face_center"])[:2]
                    prev = self._placed.get(ob.id)
                    self._placed[ob.id] = (face, float(stamp))
                    tol = max(c["map_place_tol_m"], c["map_place_tol_rel"] * ob.info["range_m"])
                    stale = prev is None or float(stamp) - prev[1] > c["max_frame_gap_s"]
                    if stale or np.linalg.norm(face - prev[0]) > tol:
                        dbg.setdefault("held_back", []).append(int(ob.id))
                        continue
                    out.obstacles.append(ob)
                tr.set_attr("obs_id", obs_ids)
        # A raw (map-free) box goes into sortie-long memory: when the map comparison ran and explains most of the
        # emergency cluster (a mapped surface), the box is not emitted. The emergency FLAG is unaffected (DN-25).
        if raw_pending:
            explained = False
            if use_map:
                bgm = background[emerg_feats]
                explained = bool(bgm.mean() >= c["raw_explained_frac"]) if len(bgm) else False
            if explained:
                dbg["raw_suppressed"] = "map_explained"
            else:
                T_map_cam = np.asarray(T_map_body, dtype=np.float64) @ self.T_body_cam
                P = X_cam[emerg_feats] @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
                out.obstacles.append(self._box(P, T_map_cam, None, stamp, "looming_raw", emerg_feats,
                                               float(np.median(contact[emerg_feats]))))
        return done("ok")

    # ----------------------------------------------------------------------------------------------------------
    def _background_above(self, mem, background, uv, Z, zc, z_map):
        """Features above the cluster in the image (within its column span) that are map-consistent AND measure
        clearly BEHIND the cluster (their own small-neighbourhood depth), so a mis-classified piece of the obstacle
        itself can never close its top (DN-29)."""
        u0, u1 = np.percentile(uv[mem, 0], [10, 90])
        v_top = float(uv[mem, 1].min())
        v_top_robust = float(np.percentile(uv[mem, 1], 10))  # for the out-of-view decision only (DN-29)
        above = (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] < v_top)
        above[mem] = False
        f = self.cfg["top_bg_depth_factor"]
        # anything above that measures NEAR (its own depth within f x the cluster's) is more obstacle: only
        # background rays above the topmost such feature can bound the top
        # features above measuring NEAR (own depth within top_near_depth_factor x the cluster's) are more obstacle:
        # rays above the 2nd-topmost of them only can bound the top (one noisy depth cannot cut the column)
        clearly_near = above & np.isfinite(Z) & (Z <= self.cfg["top_near_depth_factor"] * zc)
        vn = np.sort(uv[clearly_near, 1])
        k = int(self.cfg["top_near_min_features"])
        v_lim = float(vn[k - 1]) if len(vn) >= k else v_top
        # a ray whose own depth lies clearly beyond the obstacle (and its assumed depth) AND on the mapped surface
        # passed over it. Mapped = the map test, or its own depth agreeing with its map depth (just above the
        # obstacle's edge the wide map neighbourhood spreads excess onto the background); the obstacle's own top
        # surface, seen from above, is far but not mapped and never counts.
        far = np.isfinite(Z) & (Z > max(f * zc, zc + self.cfg["assumed_depth_m"]))
        on_map = background | (np.isfinite(z_map) & (np.abs(Z - z_map) <= self.cfg["top_bg_map_tol"] * z_map))
        cand = above & far & on_map & (uv[:, 1] < v_lim)
        return np.nonzero(cand)[0], v_top_robust, int(clearly_near.sum())

    def _box(self, P, T_map_cam, top_info, stamp, source, mem, ttc):
        """Axis-aligned map box from front-face points P (map). top_info = (bg idx, n_now, zc) or None."""
        c = self.cfg
        D = float(c["assumed_depth_m"])
        fwd = T_map_cam[:3, 2].copy()
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        back = P + fwd * D
        allp = np.vstack([P, back])
        pad = float(c["edge_pad_m"])
        lo = allp.min(axis=0) - pad
        hi = allp.max(axis=0) + pad
        feat_top = float(P[:, 2].max())
        lo[2] = self.floor_z if self.floor_z is not None else float(P[:, 2].min()) - pad
        top_seen = False
        top_bound = None
        if top_info is not None:
            bg, n_now, zc, v_top, n_near = top_info
            if len(bg) >= c["top_min_features"]:
                r = np.column_stack([n_now[bg], np.ones(len(bg))]) @ T_map_cam[:3, :3].T   # unit camera-z rays
                cz = T_map_cam[2, 3]
                hb = np.minimum(cz + r[:, 2] * zc, cz + r[:, 2] * (zc + D))                 # min over the depth span
                hb = np.sort(hb[hb >= feat_top])
                if len(hb) >= c["top_min_features"]:
                    # the k-th lowest ray, not the lowest: one mis-classified feature cannot lower the top
                    top_bound = float(hb[int(c["top_min_features"]) - 1])
                    top_seen = bool(hb.max() >= top_bound + c["top_clear_m"])
        top_kind = "seen" if top_seen else "out_of_view"
        if top_seen:
            hi[2] = top_bound
        elif top_bound is not None:
            top_kind = "seen_no_clearance"              # background above, but not enough room to fly over
            hi[2] = top_bound
        elif top_info is not None and top_info[3] > c["top_edge_px"] and top_info[4] < c["top_near_min_features"]:
            # in view, nothing NEAR measured above it, but too few clear background rays to bound the top (DN-29)
            top_kind = "nothing_near_above"
            hi[2] = feat_top + c["top_unseen_pad_m"]
        else:
            hi[2] = self.ceiling_z if self.ceiling_z is not None else feat_top + D
        if self.ceiling_z is not None:
            hi[2] = min(hi[2], self.ceiling_z)
        oid = self._new_obs_id(self.tracker.attr("obs_id")[mem], stamp)
        cam_pos = T_map_cam[:3, 3]
        face = 0.5 * (P.min(axis=0) + P.max(axis=0))
        return Obstacle(id=oid, center=(lo + hi) / 2, half_size=(hi - lo) / 2, top_seen=top_seen,
                        first_seen=self._first_seen[oid], last_seen=float(stamp), source=source,
                        confidence=float(min(1.0, len(mem) / 20.0)),
                        info={"n_features": int(len(mem)), "ttc_s": float(ttc),
                              "range_m": float(np.linalg.norm((face - cam_pos)[:2])),
                              "face_center": face, "top_bound": top_bound, "top_kind": top_kind})
