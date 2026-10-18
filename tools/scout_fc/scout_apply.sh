#!/bin/bash
# Apply ONE approved Scout change set, refusing unless the FC on the bridge IS the Scout's board.
# Git Bash, with com_bridge.ps1 already running (logs every transmitted byte):
#     tools/scout_fc/scout_apply.sh <bench dir> hardware/scout/PARAM_CHANGESET_S1.json <go sha256> [--dry-run]
# Another FC with the same USB ID can appear on the same COM port, so the board is checked by
# its UID (AUTOPILOT_VERSION, read-only) against the change set's "board_uid" before anything else.
# The write itself is the bench's fc_apply_changeset.py (go-token = file sha256, drift check, read-back,
# post-dump diff; never reboots).
set -o pipefail
BENCH="${1:?bench dir (fc_readonly.py, fc_apply_changeset.py, pylib/)}"; CS="${2:?change set json}"; GO="${3:?go sha256}"; shift 3
die() { echo "scout_apply: $*" >&2; exit 2; }
[ -f "$CS" ] || die "no change set $CS"
CSW=$(cygpath -w "$(realpath "$CS")")
[ -f "$BENCH/fc_apply_changeset.py" ] || die "no fc_apply_changeset.py in $BENCH"
want_uid=$(py -3.9 -c "import json,sys; print(json.load(open(sys.argv[1]))['board_uid'])" "$CSW") || die "no board_uid in $CS"
probe="$BENCH/cap/uidcheck_$(date -u +%H%M%S).json"
( cd "$BENCH" && MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONPATH="pylib;." py -3.9 fc_readonly.py probe --seconds 2 --json-out "$(cygpath -w "$probe")" >/dev/null 2>&1 ) || die "probe failed (bridge up?)"
got_uid=$(py -3.9 -c "import json,sys; print(json.load(open(sys.argv[1]))['autopilot_version']['uid'])" "$(cygpath -w "$probe")") || die "no AUTOPILOT_VERSION"
[ "$got_uid" = "$want_uid" ] || die "REFUSED: FC uid $got_uid is not the Scout's ($want_uid). Wrong board on the cable?"
echo "scout_apply: board uid $got_uid = Scout. Running fc_apply_changeset $*"
out="$BENCH/cap/apply_$(basename "$CS" .json)_$(date -u +%H%M%S)"
cd "$BENCH" && MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONPATH="pylib;." py -3.9 fc_apply_changeset.py \
  --changeset "$CSW" --go-sha256 "$GO" --out-dir "$(cygpath -w "$out")" "$@"
