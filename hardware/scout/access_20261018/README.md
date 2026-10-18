# Scout access check after wiring (2026-10-17 night → 2026-10-18, laptop on Ron's Windows hotspot)

Ron: the FC is wired to the Radxa (TELEM2 ↔ UART4, `hardware/scout/WIRING_RADXA_FC.md`), the FC is on the laptop's USB, and
the laptop runs a hotspot with the old network's name and password ("AlphaSigmaPhi", WPA2).

## FC over USB: OK (read-only)

The read-only FC bench tools (byte-logging COM bridge + read-only MAVLink client; pymavlink copied fresh from WSL: the
old bench copy had lost its `.py` files). COM4, UID `0x3533511634343432` = the Scout's FC, PX4 1.18.0,
disarmed, rangefinder 2 cm (on the ground). Pack in: 15.56–15.58 V, 56–57 %, 0.00–0.09 A (props off; voltage read every
60 s by `batt_monitor.sh`). `uxrce_dds_client status`: **running, disconnected, transport serial** (no agent yet);
`UXRCE_DDS_CFG` 102 (TELEM2), `SER_TEL2_BAUD` 921600. The FC side of the companion link is ready.
Transmit logs (every byte sent) audit clean: `tx_scout_access_20261018{,b,c,d}.bin` = HEARTBEAT 160, COMMAND_LONG
(REQUEST_MESSAGE) 1, SERIAL_CONTROL (whitelisted shell) 4; `non_whitelisted: []`, no reboots. Probe: `probe_scout.json`.

## Radxa on Wi-Fi: NOT reachable

The Radxa boots (power LED + heartbeat LED, Ron) but never even ASSOCIATES with the hotspot. Windows' WLAN-AutoConfig log
shows every association attempt (events 20019/20020): only Ron's iPhone and Watch, never the Radxa, over ~1 h. A wrong
password would still show an association, so this is "does not see / does not try", not a credential failure.

| Test | Result |
|---|---|
| Hotspot band 2.4 GHz, WPA2, SSID AlphaSigmaPhi (WinRT query) | settings match the old network; 0 Radxa attempts |
| Laptop's own Wi-Fi on campus 5 GHz ch 144 (hotspot may share that radio channel) | moved the laptop to 2.4 GHz only (Killer "Wireless Mode" 05 - 11 b/g/n; `tools/laptop_wifi_band.ps1`), laptop on 2.4 GHz ch 6: still 0 attempts |
| Battery replug (Radxa reboots with the hotspot already up) | 0 attempts |
| Radxa powered from the laptop's USB-C instead of the UBEC | 0 attempts in 10 min |

Its Wi-Fi worked on 2026-10-15 (router AlphaSigmaPhi, 2.4 GHz ch 1, 10.1.10.77, `ssh scout`); later that day, powered from
a laptop USB port, it was not on Wi-Fi (NEEDS_RON R9). Untested: a 5 V ≥ 2 A charger / power bank (none at hand), the original
router. Next: look inside it (micro-HDMI + USB keyboard, or a USB-serial adapter on pins 8/10 = the debug console):
`nmcli dev wifi list`, `nmcli con show`, `journalctl -u NetworkManager`, `dmesg | grep -i -e aic -e wlan` (driver / firmware).

Laptop setting changed for this test (Ron ran it): Killer adapter "Wireless Mode" = 05 - 11 b/g/n. Revert with
`powershell -ExecutionPolicy Bypass -File tools\laptop_wifi_band.ps1 -Revert` (admin prompt).

## Resolved (2026-10-18, ~01:10 local)

**Cause of the dead Wi-Fi:** `/boot/armbianEnv.txt` on the SD card read as 272 NUL bytes. An edit (most likely adding the
UART4 overlay) was cut off by a power loss before ext4 wrote the data (the root fs mounts with `data=writeback`). U-Boot
cannot replay an ext4 journal, so it read the zeroed file, booted with its default `rootdev=/dev/mmcblk0p1` and no
`fdtfile`, and Linux never came up: heartbeat LED, no Wi-Fi. Found by reading the card off the laptop with a new read-only
extractor (`tools/ext4_extract.py`, tested: 604 files byte-identical on a WSL ext4 image; `tools/read_radxa_sd.bat`,
`tools/inspect_radxa_sd.bat`) and comparing with the untouched Armbian image (UUID 2bae8c0f-… matches).

**Fix (nothing written by hand):** usbipd-win (Ron installed it) passed the SD reader to WSL (`usbipd bind/attach --busid
1-17`); mounting the card there replayed its journal ("recovery complete") and the file came back valid (the original +
Armbian's `usbstoragequirks` line, saved 10-16 19:15). `e2fsck -fn`: clean. Card back in → joined the hotspot as
`scout.mshome.net` (192.168.137.216); `ssh scout` from Windows (key copied to `C:\Users\Ron\.ssh`).

**FC link:** the UART edit redone safely over ssh (backup `armbianEnv.txt.bak-20261018`, `sync`, re-read after dropping
caches, clean `systemctl reboot`). UART4-M1 (pins 16 / 18) stayed silent (rx 0, no framing errors, no edges on either
pin). Watching every free header pin found the FC's TX on **pin 3**: the wiring is on UART3-M0 (pins 3 / 5). With
`overlays=rk3568-uart3-m0` (backup `armbianEnv.txt.bak-uart4-20261018`), `/dev/ttyS3` at 921600 receives the FC's
uXRCE-DDS ping frames: 138 B / 6 s, `7e 01 00 10 00 80 … cf 12` every ~1 s. **Radxa → FC is not verified yet** (needs
MicroXRCEAgent on the Radxa to answer; not installed). FC tx log `tx_scout_access_20261018e.bin`: 30 HEARTBEAT, clean.
