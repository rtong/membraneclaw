#!/bin/bash
# Run inside WSL if `systemctl` fails with:
#   "Failed to connect to system scope bus via local transport"
#
# Some WSL instances don't reliably start dbus on boot (more likely after a
# suspend/resume than a genuine cold boot). Without dbus, systemctl can't
# talk to systemd at all, which breaks anything systemd-managed —
# tailscaled included, which silently takes out any Tailscale Serve/Funnel
# URL until this is fixed.
#
# This is a per-boot manual fix, not a permanent one. The qwen-rtx
# deployment's start.sh already runs this check automatically before
# launching, so this script is mainly useful standalone, before you've
# cloned/run that deployment yet.
set -e
if [ ! -S /run/dbus/system_bus_socket ]; then
  echo "dbus not running, fixing..."
  sudo mkdir -p /run/dbus
  sudo dbus-daemon --system --fork
else
  echo "dbus already running."
fi
sudo systemctl start tailscaled
systemctl status tailscaled --no-pager | head -5
