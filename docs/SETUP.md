# Setup steps for Ron

Two one-time jobs: the laptop's Ethernet for the L2, and the Scout's Radxa SD card.

## 1. Laptop Ethernet for the L2 (about 2 minutes)

The L2 sends its data to `192.168.1.2`, port 6201 (DH-04). The laptop takes that address on its
wired port, so the L2 itself is never reconfigured. The wired port is the adapter Windows calls
**"Ethernet"** (Killer E2500). Your home Wi-Fi is on 10.1.10.x, so nothing clashes.

Open **PowerShell as Administrator** (Start → type "PowerShell" → right-click → Run as administrator)
and paste these two commands:

```powershell
netsh interface ipv4 set address name="Ethernet" static 192.168.1.2 255.255.255.0
New-NetFirewallRule -DisplayName "Noxen L2 lidar (UDP 6201)" -Direction Inbound -Protocol UDP -LocalPort 6201 -RemoteAddress 192.168.1.62 -Action Allow -Profile Any
```

- The first gives the wired port the fixed address. No gateway, so it never touches your internet.
  (Not `New-NetIPAddress`: it fails with "Inconsistent parameters PolicyStore PersistentStore and
  Dhcp Enabled" while the port is still on DHCP, which happened on 2026-10-15. `netsh` turns DHCP
  off and sets the address in one step, cable or no cable.)
- The second lets the L2's packets in, and only from the L2's own address (192.168.1.62).
  `-Profile Any` matters: a cable with no router shows up as an "unidentified" Public network.

Then plug the L2 into the laptop's Ethernet port and power it. Tell me, and I run the capture.

**To undo later** (e.g. to use that port on a normal wired network), in an admin PowerShell:

```powershell
netsh interface ipv4 set address name="Ethernet" source=dhcp
Remove-NetFirewallRule -DisplayName "Noxen L2 lidar (UDP 6201)"
```

## 2. Radxa Zero 3W SD card (about 20 minutes)

**What to flash and why.** The Pi's image will not boot on the Radxa (different chip), so it gets
its own: **Armbian, Debian 13 "Trixie" Minimal (CLI), vendor kernel 6.1**. That is the image
Armbian hand-picks as stable for this board. Armbian no longer offers Ubuntu 24.04 for it (only
26.04), and that's fine: the Scout's onboard computer doesn't need ROS (DH-07). It only relays
between the flight controller and the laptop and streams the camera.

You need: the microSD card, an SD reader on the laptop, and the Wi-Fi name and password the Scout
should join (your home Wi-Fi for now; we add the event network later).

1. **Install Armbian Imager** from https://www.armbian.com/radxa-zero3/ (the "Download with Armbian
   Imager" button). It's Armbian's official flashing tool, like Raspberry Pi Imager.
2. Open it and pick **Radxa ZERO 3** as the board, then the **Debian 13 (Trixie) Minimal / CLI** image
   with the **vendor** kernel (6.1.x).
3. Pick the SD card as the target. **Double-check it's the SD card**, not another drive: flashing
   erases the target.
> **2026-10-15: Ron's Armbian Imager showed NO customization step**, so the card was written stock
> and the first login was done by hand (screen + keyboard, below). Step 4 is kept for a future
> re-flash, in case a newer Imager shows the settings (possibly behind a gear icon).

4. If your Imager has **customization / first-boot settings** (a gear or "Customize" step), set:
   - **Hostname:** `scout`
   - **Username:** `noxen` (same as the Pi). **Password:** your choice. I never type passwords, so
     you set it here.
   - **Wi-Fi:** SSID + password, **country US**.
   - **SSH public key:** paste this whole line (it's the laptop's key for the Scout, made for this;
     the private half stays in WSL at `~/.ssh/id_ed25519_scout`):
     ```
     ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILGniCPFJVe4fGX7NKtyNN14JjPxDZSiUu9ccq7HnWcm noxen-laptop-to-scout
     ```
     (also saved in this repo as `config/ssh/scout_laptop_key.pub`)
   - **Timezone:** your local one.
5. Write the card, put it in the Radxa, and power the Radxa (USB-C). The first boot takes a few
   minutes (it expands the filesystem and applies the settings).
6. Tell me it's booted. I find it on your network (it should appear as `scout` on 10.1.10.x) and log
   in with the key. You don't need to find the IP yourself.

**What actually worked on 2026-10-15: settings written INTO the image, then flashed.** Armbian Imager
2.x showed no first-boot settings, and `wsl --mount` refuses a disk behind a USB SD reader
(`Wsl/Service/AttachDisk/MountDisk/0x8007000f`). So, all from this repo's `tools/scout_sdcard/`:

1. `bash fetch_image.sh` (WSL): downloads the Armbian image, checks Armbian's sha256, decompresses
   it to `~/noxen_dh26/images/`.
2. `wsl -u root bash image_open.sh <img>`: loop-mounts it and runs `write_presets.sh` (presets,
   hostname `scout`, the laptop's SSH key for root, a one-shot `provisioning.sh` that gives `noxen`
   the key on first boot).
3. Ron: `wsl -u root bash fill_secrets.sh /mnt/scout_img` (Wi-Fi name + key, root + noxen passwords).
4. `wsl -u root bash image_close.sh <img>`: detaches it. Flash it straight from
   `\\wsl.localhost\Ubuntu-24.04\home\ron\noxen_dh26\images\<img>`: no 2.3 GB copy onto C:, which ran
   out of space on 2026-10-15 and briefly made WSL's disk read-only.
5. Ron flashes that `.img` with Raspberry Pi Imager → Choose OS → **Use custom**. Answer **No** to
   "apply OS customisation settings?" (those are Raspberry Pi OS only).
6. Card into the Radxa, USB-C charger power, wait ~5 min.

**First login by hand (no presets, or the presets were ignored).** The Zero 3W has no Ethernet, so
without Wi-Fi presets it is unreachable until someone logs in once:

1. Mini-HDMI to a screen; a USB keyboard into the USB-C port NOT used for power (USB-C-to-A adapter).
   Power from a USB-C phone charger (5 V, ≥ 2 A) rather than the Scout battery: the battery makes
   the ESCs live, and the FC firmware is not PX4 yet.
2. Log in as `root` / `1234` (Armbian default). Set a new root password, then create user `noxen`.
3. Join Wi-Fi with `nmtui` → Activate a connection, if first login didn't offer it.
4. `hostnamectl set-hostname scout`, then `ip -4 addr show wlan0` for the address.
5. From the laptop, install the key once (Ron types the `noxen` password; Claude never does):
   `ssh-copy-id -i ~/.ssh/id_ed25519_scout.pub noxen@<address>` in WSL.

If there's no screen, a USB-to-serial adapter (CP2104 or CH340, not PL2303) on the board's debug
pins works too.

**Not part of this step:** flashing PX4 onto the MicoAir flight controller (you said you'd do it).
Use PX4 1.18.0 RC `fca3df86` so the reference stability parameters carry over. Tell
me when you're ready and I'll find the matching MicoAir 743 AIO firmware build and write the steps.
