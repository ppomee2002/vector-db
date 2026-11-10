#!/usr/bin/env bash
# dm-delay volume for Qdrant: loop file (direct-io) -> dm-delay -> ext4 at ~/gran/qvol-d (copy of ~/gran/qvol).
#   delay_dev.sh setup        # create + copy (Qdrant must be stopped)
#   delay_dev.sh set <ms>     # change read delay (Qdrant must be stopped; suspends the device)
#   delay_dev.sh teardown
set -euo pipefail
G=$HOME/gran; IMG=$G/delay.img; MNT=$G/qvol-d; DEV=gran-delay
S="sudo -n"
table() { local loop sz; loop=$($S losetup -j "$IMG" | cut -d: -f1 | head -1); sz=$($S blockdev --getsz "$loop")
          echo "0 $sz delay $loop 0 $1 $loop 0 0"; }
case $1 in
  setup)
    [ -e /dev/mapper/$DEV ] && { echo "exists"; exit 0; }
    [ -f "$IMG" ] || truncate -s 12G "$IMG"
    $S losetup -f --direct-io=on "$IMG"
    table 0 | $S dmsetup create $DEV
    $S mkfs.ext4 -q -F /dev/mapper/$DEV
    mkdir -p "$MNT"; $S mount /dev/mapper/$DEV "$MNT"
    $S cp -a "$G/qvol/." "$MNT/"; sync; echo "setup ok: $($S dmsetup table $DEV)";;
  set)
    $S dmsetup suspend $DEV; table "${2:?ms}" | $S dmsetup reload $DEV; $S dmsetup resume $DEV
    echo "delay now: $($S dmsetup table $DEV)";;
  teardown)
    $S umount "$MNT" || true; $S dmsetup remove $DEV || true
    for l in $($S losetup -j "$IMG" | cut -d: -f1); do $S losetup -d "$l"; done; echo "torn down (image kept: $IMG)";;
esac
