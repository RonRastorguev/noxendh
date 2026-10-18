#!/bin/bash
# Download + verify + decompress the Armbian image for the Scout's Radxa, in WSL (never on /mnt/c).
#     bash tools/scout_sdcard/fetch_image.sh
# Output: ~/noxen_dh26/images/<name>.img (raw, ready for write_presets via a loop device).
set -o pipefail
NAME=Armbian_26.8.1_Radxa-zero3_trixie_vendor_6.1.115_minimal.img.xz
URL=https://dl.armbian.com/radxa-zero3/archive/$NAME
D=$HOME/noxen_dh26/images
mkdir -p "$D" && cd "$D" || exit 2
[ -f "$NAME" ] || curl -fL --retry 3 -o "$NAME.part" "$URL" && { [ -f "$NAME" ] || mv "$NAME.part" "$NAME"; } || { echo "download failed"; exit 1; }
curl -fsSL -o "$NAME.sha" "$URL.sha" || { echo "checksum file download failed"; exit 1; }
want=$(awk '{print $1}' "$NAME.sha")
got=$(sha256sum "$NAME" | awk '{print $1}')
echo "sha256 published: $want"
echo "sha256 computed:  $got"
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "CHECKSUM MISMATCH"; exit 1; }
[ -f "${NAME%.xz}" ] || xz -dk "$NAME" || exit 1
ls -l "$D"
echo "OK ${D}/${NAME%.xz}"
