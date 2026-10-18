"""Camera relocalization in a prior lidar map (ARCHITECTURE §8.2–8.3). Pure Python: numpy + OpenCV, no ROS.

Sensor-agnostic on purpose (DH-09 style): the camera, its mount and every threshold come from config, so the same
code serves the Scout today and any camera-carrying platform later. Poses are 4x4 numpy arrays `T_a_b` that map
points from frame b into frame a (so `T_map_cam` is the camera's pose in the map). Camera frames are OpenCV
optical (x right, y down, z forward).

Modules:
    geometry      pose helpers (inverse, rotation error, quaternions, look-at, interpolation)
    camera        pinhole (+ OpenCV distortion) camera model; nominal model from a field of view
    features      ORB extraction with contrast normalization and grid bucketing
    depth_render  z-buffer rendering of map points into a camera (also what §8.4 step 3 needs)
    landmarks     landmark database: build from keyframes + map cloud, save / load
    orb_pnp       per-frame ORB matching -> solvePnPRansac
    apriltag      tag-based PnP fallback (§8.3)
    localizer     the loc_ok state machine on top of the estimators; the one entry point for a node
    config        YAML config -> Localizer
    synthetic_room  textured synthetic room, renderer and trajectories for tests (no hardware data yet)
"""
