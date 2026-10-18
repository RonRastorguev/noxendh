# unitree_l2 test fixtures — provenance

Real captures from the bench L2 (Unitree L2, hw 2.2.1.1, sw 2.8.11.1, "YS-L2"), recorded on the Pi's
`eth0` with `tcpdump -ni eth0 -s 0 udp port 6201`. Bytes are copied verbatim (pcap global header + whole
records); nothing is synthesised or edited.

| File | Source | Slice | What it contains / proves |
|---|---|---|---|
| `l2_real_wide_5s.pcap` | 2026-09-30 bench capture (IMU off, wide FOV) | 5 s | 1 068 point packets; the timebase test (`test_real_pcap_timebase`) |
| `l2_real_imu_coalesced_3s.pcap` | `~/noxen_bench/l2/window2/imu_check_20s.pcap` on the Pi (2026-10-06 22:23Z, battery window 2, IMU just enabled, mode 1), sha256 `07d83848…6717fff` | capture seconds 5.0–8.0 (relative to the first record), 1 370 datagrams, sha256 `f2760eeb…a75b5c0b` | 34 datagrams carrying an IMU frame (80 B) AND a point frame (1 044 B) together (1 124 B UDP payload); 753 IMU + 651 point frames. `test_real_pcap_coalesced_imu_point` |

The slice was cut with a byte-copying pcap slicer (record timestamps relative to the first record,
`5.0 <= t < 8.0`); the full 20 s capture stays on the Pi.
