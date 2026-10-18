#!/bin/bash
# Prepare a freshly flashed Armbian card for the Scout's Radxa BEFORE its first boot (docs/SETUP.md §2).
# Run as root in WSL with the card's root filesystem mounted:
#     wsl -u root bash tools/scout_sdcard/write_presets.sh /mnt/scout_sd
# Writes everything EXCEPT secrets: Armbian's first-boot presets (docs.armbian.com/user-guide/autoconfig),
# the hostname, and a one-shot /root/provisioning.sh that installs the laptop's SSH key for `noxen`.
# Secrets (Wi-Fi name + key, passwords) are added by fill_secrets.sh, typed by Ron, never by an agent.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="${1:?usage: write_presets.sh <card rootfs mount>}"
KEY_FILE="$HERE/../../config/ssh/scout_laptop_key.pub"
die() { echo "write_presets: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (wsl -u root ...)"
[ -f "$R/etc/armbian-release" ] || die "$R does not look like an Armbian root filesystem"
grep -q '^BOARD=radxa-zero3' "$R/etc/armbian-release" || die "not a radxa-zero3 image: $(grep ^BOARD= "$R/etc/armbian-release")"
[ -f "$R/root/.not_logged_in_yet" ] || die "$R/root/.not_logged_in_yet missing: this card has already been booted (first login done)"
KEY="$(tr -d '\r\n' < "$KEY_FILE")"
case "$KEY" in ssh-ed25519\ *) ;; *) die "bad key file $KEY_FILE" ;; esac

# 1. first-boot presets (non-secret part). Kept: the image's original flag file, as a .orig copy.
[ -f "$R/root/.not_logged_in_yet.orig" ] || cp -p "$R/root/.not_logged_in_yet" "$R/root/.not_logged_in_yet.orig"
cat > "$R/root/.not_logged_in_yet" <<'EOF'
# Noxen Scout first-boot presets (noxen-dh26 tools/scout_sdcard). Secrets are appended by fill_secrets.sh.
PRESET_NET_CHANGE_DEFAULTS="1"
PRESET_NET_ETHERNET_ENABLED="0"
PRESET_NET_WIFI_ENABLED="1"
PRESET_NET_WIFI_COUNTRYCODE="US"
PRESET_CONNECT_WIRELESS="n"
PRESET_NET_USE_STATIC="0"
SET_LANG_BASED_ON_LOCATION="n"
PRESET_LOCALE="en_US.UTF-8"
PRESET_TIMEZONE="Etc/UTC"
# ^ REQUIRED for a headless boot: without it armbian-firstlogin ran the interactive tzselect on tty1 and
#   waited there forever (2026-10-15): no provisioning, and this file (with secrets) never deleted.
PRESET_USER_NAME="noxen"
PRESET_DEFAULT_REALNAME="Noxen"
PRESET_USER_SHELL="bash"
# --- secrets below (fill_secrets.sh) ---
EOF
chmod 600 "$R/root/.not_logged_in_yet"

# 2. hostname
echo scout > "$R/etc/hostname"
if grep -q 'radxa-zero3' "$R/etc/hosts"; then
  sed -i 's/radxa-zero3/scout/g' "$R/etc/hosts"
elif ! grep -qw scout "$R/etc/hosts"; then
  echo "127.0.1.1 scout" >> "$R/etc/hosts"
fi

# 3. SSH key: for root now (backstop: reachable even if user creation fails), for noxen via provisioning.sh
install -d -m 700 "$R/root/.ssh"
grep -qxF "$KEY" "$R/root/.ssh/authorized_keys" 2>/dev/null || echo "$KEY" >> "$R/root/.ssh/authorized_keys"
chmod 600 "$R/root/.ssh/authorized_keys"
cat > "$R/root/provisioning.sh" <<EOF
#!/bin/bash
# Noxen: runs once as root after Armbian's first login (autoconfig docs). Installs the laptop's key for noxen.
install -d -m 700 -o noxen -g noxen /home/noxen/.ssh
grep -qxF '$KEY' /home/noxen/.ssh/authorized_keys 2>/dev/null || echo '$KEY' >> /home/noxen/.ssh/authorized_keys
chown noxen:noxen /home/noxen/.ssh/authorized_keys
chmod 600 /home/noxen/.ssh/authorized_keys
echo "noxen provisioning done \$(date -Is)" >> /root/noxen_provisioning.log
EOF
chmod 700 "$R/root/provisioning.sh"

echo "write_presets: OK on $R ($(grep -E '^(VERSION|BOARD|BRANCH|DISTRIBUTION_CODENAME)=' "$R/etc/armbian-release" | tr '\n' ' '))"
echo "next: Ron runs fill_secrets.sh (Wi-Fi + passwords)"
