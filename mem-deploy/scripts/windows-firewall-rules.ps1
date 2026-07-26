# Run in an ELEVATED (Administrator) PowerShell on the Windows host.
#
# Why this is needed: in WSL2 mirrored networking mode, traffic to a WSL
# service's own port (even self-directed, even via the tailscale IP) can
# route through a Hyper-V-managed relay path that Windows' firewall filters
# per-port. Without these rules, ports can appear to work for some services
# (e.g. sshd on 22) but silently hang for others (e.g. a freshly-bound app
# port), which is very confusing to debug from the WSL side alone.
#
# Adjust the port list to whatever the deployment actually uses.
# 8000 = vLLM's own port (see qwen-rtx/profile.env's PORT)
# 443  = tailscale serve/funnel's HTTPS front

netsh advfirewall firewall add rule name="WSL vLLM 8000" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="WSL Tailscale Serve 443" dir=in action=allow protocol=TCP localport=443
