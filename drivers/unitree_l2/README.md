# unitree_l2

Read-only Unitree L2 support, no vendor code (DH-02).

| File | Origin | Role |
|---|---|---|
| `l2_protocol.py` | reused verbatim from our earlier L2 work (organizer-approved, DH-02) | wire format, calibration math, coalesced-datagram split, CRC, timebase classes |
| `test_l2_protocol.py` | same | 95 offline checks, incl. the real-sensor fixtures |
| `bench/l2_pcap.py` | same | read / analyze / replay raw L2 captures |
| `test_fixtures/*.pcap` | same (real bench captures; provenance in `test_fixtures/README.md`) | real L2 data for tests |
| `receiver.py` | new in this repo | UDP accumulation for a fixed sensor (the table demo) |

Nothing here ever sends to the sensor. Configuring the L2 (work mode, IP) is out of scope; the laptop
takes the L2's default host address instead (DH-04). Never run Unitree's `unilidar_sdk2` against it.

Comments in `l2_protocol.py` refer to the vendor protocol notes and the earlier bench measurements by name.
