#!/usr/bin/env bash
# free_ram.sh — release RAM on Jetson Nano (2GB) before heavy work.
# Run before trtexec build (tools/swap_model.sh) or edge_pipeline.py.
#
# Stops non-essential desktop / IoT services and drops OS caches.
# Idempotent: safe to re-run. Each stop is best-effort (`|| true`) so a
# missing service does not abort the rest.
#
# Permanent GUI removal (recommended for ssh-only use):
#   sudo systemctl set-default multi-user.target && sudo reboot

set -u

if [ "$(id -u)" -ne 0 ]; then
  echo "rerun with sudo: sudo bash tools/free_ram.sh" >&2
  exit 1
fi

echo "[free_ram] baseline:"
free -h
echo

# 1. Display manager + X session bits (LXDE: Xorg, lxpanel, compton).
#    On an ssh-only Jetson these are pure waste. Stopping the display
#    manager cascades into Xorg + the LXDE session.
for svc in lightdm gdm3 gdm display-manager; do
  systemctl stop "$svc" 2>/dev/null && echo "  stopped $svc" || true
done
# Belt-and-braces in case the user session lingered after display-manager.stop:
pkill -9 -x lxpanel 2>/dev/null && echo "  killed lxpanel" || true
pkill -9 -x compton 2>/dev/null && echo "  killed compton" || true

# 2. Tegra CSI camera daemon. We feed RTSP, so libargus is dead weight.
systemctl stop nvargus-daemon 2>/dev/null && echo "  stopped nvargus-daemon" || true

# 3. mDNS / .local discovery. avahi-daemon was at 8.6% CPU on this host.
#    Safe to stop unless something on the LAN resolves <host>.local.
systemctl stop avahi-daemon avahi-daemon.socket 2>/dev/null && echo "  stopped avahi-daemon" || true

# 4. Ubuntu desktop bloat (telemetry, printing, BT, modem, snap, auto-upgrade).
for svc in cups cups-browsed bluetooth whoopsie kerneloops ModemManager \
           unattended-upgrades snapd snapd.socket; do
  systemctl stop "$svc" 2>/dev/null && echo "  stopped $svc" || true
done

# 5. (opt-in) Docker. Uncomment if you don't use Docker on this Jetson —
#    containerd + dockerd together save ~80–150 MB.
# systemctl stop docker 2>/dev/null && echo "  stopped docker" || true
# systemctl stop containerd 2>/dev/null && echo "  stopped containerd" || true

# 6. Drop OS caches (page / dentries / inodes). Dirty pages flush first.
sync
echo 3 > /proc/sys/vm/drop_caches
echo "  dropped OS caches"

echo
echo "[free_ram] after:"
free -h
echo
echo "If you only ssh in, make this permanent across reboots:"
echo "  sudo systemctl set-default multi-user.target  &&  sudo reboot"
