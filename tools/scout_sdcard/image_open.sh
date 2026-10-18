#!/bin/bash
# Attach the downloaded Armbian image, mount its root filesystem and write the non-secret presets.
#     wsl -u root bash tools/scout_sdcard/image_open.sh <path/to/image.img>
# Leaves it mounted at /mnt/scout_img for fill_secrets.sh; image_close.sh detaches it.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="${1:?usage: image_open.sh <image.img>}"
M=/mnt/scout_img
die() { echo "image_open: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root (wsl -u root ...)"
[ -f "$IMG" ] || die "no image at $IMG"
mountpoint -q "$M" && die "$M already mounted (run image_close.sh first)"
LOOP=$(losetup -fP --show "$IMG") || die "losetup failed"
echo "$LOOP" > /run/scout_img.loop
for _ in 1 2 3 4 5; do [ -b "${LOOP}p1" ] && break; sleep 0.5; done
[ -b "${LOOP}p1" ] || { losetup -d "$LOOP"; die "no partition device ${LOOP}p1"; }
mkdir -p "$M"
mount "${LOOP}p1" "$M" || { losetup -d "$LOOP"; die "mount failed"; }
bash "$HERE/write_presets.sh" "$M" || die "write_presets failed (still mounted; run image_close.sh)"
echo "image_open: ready at $M ($LOOP)"
