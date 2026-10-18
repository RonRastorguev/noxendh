#!/bin/bash
# Unmount and detach the image prepared by image_open.sh, then copy it to Windows for flashing.
#     wsl -u root bash tools/scout_sdcard/image_close.sh <path/to/image.img> [windows dest dir, as /mnt/c/...]
# Without a dest dir nothing is copied: flash straight from \\wsl.localhost\Ubuntu-24.04\<image path>
# (saves 2.3 GB on C:, which ran out on 2026-10-15).
set -o pipefail
IMG="${1:?usage: image_close.sh <image.img> [dest dir]}"
DEST="$2"
M=/mnt/scout_img
die() { echo "image_close: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root (wsl -u root ...)"
if mountpoint -q "$M"; then
  grep -q '^PRESET_USER_PASSWORD=' "$M/root/.not_logged_in_yet" || die "secrets not filled in yet (fill_secrets.sh); left mounted"
  sync; umount "$M" || die "umount failed"
fi
[ -f /run/scout_img.loop ] && losetup -d "$(cat /run/scout_img.loop)" && rm -f /run/scout_img.loop
if [ -z "$DEST" ]; then
  echo "image_close: detached. Flash: \\\\wsl.localhost\\Ubuntu-24.04$(echo "$IMG" | tr / '\\')"
  exit 0
fi
mkdir -p "$DEST"
out="$DEST/scout_$(basename "$IMG")"
cp "$IMG" "$out" && echo "image_close: flash this file: $out ($(du -h "$out" | cut -f1))"
