#!/usr/bin/env bash
# Dump GT clocks / power / temp for the two B70s. CPU-side sysfs; no GPU lease.
# Usage: clocks.sh [card0|card1|both]
set -euo pipefail
which="${1:-both}"
dump_card() {
  local n="$1"
  local dev="/sys/class/drm/card${n}/device"
  echo "card${n} pci=$(basename "$(readlink -f "$dev")") power_state=$(cat "$dev/power_state")"
  for gt in gt0 gt1; do
    local f="$dev/tile0/$gt/freq0"
    echo "  $gt act=$(cat "$f/act_freq") cur=$(cat "$f/cur_freq") min=$(cat "$f/min_freq") max=$(cat "$f/max_freq") rp0=$(cat "$f/rp0_freq") rpn=$(cat "$f/rpn_freq") throttle=$(cat "$f/throttle/status" 2>/dev/null || echo n/a)"
  done
}
case "$which" in
  0|card0) dump_card 0 ;;
  1|card1) dump_card 1 ;;
  both|*) dump_card 0; dump_card 1 ;;
esac
# xe hwmon (order is not a stable card map; print both)
for h in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$h/name" 2>/dev/null || true)" = xe ] || continue
  echo "hwmon $(basename "$h") cap_uW=$(cat "$h/power1_cap") energy1=$(cat "$h/energy1_label"):$(cat "$h/energy1_input") energy2=$(cat "$h/energy2_label"):$(cat "$h/energy2_input") temp2_mC=$(cat "$h/temp2_input" 2>/dev/null || echo n/a)"
done
