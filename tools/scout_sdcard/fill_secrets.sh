#!/bin/bash
# RON runs this, in a terminal, after write_presets.sh:
#     wsl -u root bash tools/scout_sdcard/fill_secrets.sh /mnt/scout_sd
# Asks for the Wi-Fi name, the Wi-Fi password, a root password and the noxen password, and appends them to
# the card's first-boot preset file. Nothing typed is echoed or logged. Armbian stores them in plaintext on
# the card until first boot applies them (its docs say so); the preset file is consumed on first boot.
set -o pipefail
R="${1:?usage: fill_secrets.sh <card rootfs mount>}"
F="$R/root/.not_logged_in_yet"
die() { echo "fill_secrets: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root (wsl -u root ...)"
grep -q '^# --- secrets below' "$F" 2>/dev/null || die "$F was not prepared by write_presets.sh"
grep -q '^PRESET_NET_WIFI_SSID=' "$F" && die "secrets are already in $F (re-run write_presets.sh to start over)"

ok() {   # the preset file is sourced by bash inside double quotes: refuse characters that would break it
  case "$1" in *'"'*|*'$'*|*'`'*|*'\'*) echo "  not allowed: \" \$ \` or \\ (pick another or change it later)"; return 1 ;; esac
  [ -n "$1" ] || { echo "  empty"; return 1; }
}
ask() {  # ask <prompt> <secret 0|1> -> REPLY
  while true; do
    if [ "$2" = 1 ]; then read -rsp "$1: " REPLY; echo; else read -rp "$1: " REPLY; fi
    ok "$REPLY" || continue
    if [ "$2" = 1 ]; then read -rsp "  again: " again; echo; [ "$again" = "$REPLY" ] || { echo "  didn't match"; continue; }; fi
    return 0
  done
}
ask "Wi-Fi name (SSID)" 0;            ssid=$REPLY
ask "Wi-Fi password" 1;               wkey=$REPLY
ask "root password for the Radxa" 1;  rpw=$REPLY
ask "noxen password for the Radxa" 1; upw=$REPLY
{
  printf 'PRESET_NET_WIFI_SSID="%s"\n' "$ssid"
  printf 'PRESET_NET_WIFI_KEY="%s"\n' "$wkey"
  printf 'PRESET_ROOT_PASSWORD="%s"\n' "$rpw"
  printf 'PRESET_USER_PASSWORD="%s"\n' "$upw"
} >> "$F"
unset wkey rpw upw again REPLY
echo "fill_secrets: done (Wi-Fi '$ssid'). Tell Claude, which unmounts the card."
