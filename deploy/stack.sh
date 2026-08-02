#!/usr/bin/env bash
# Bring the stack up, or check it, across both hosts.
#
#   deploy/stack.sh status    what is running, and does it answer
#   deploy/stack.sh up        start anything stopped, then re-check
#   deploy/stack.sh logs      tail the last errors from every unit
#
# Run from anton. temur is reached over ssh, so the ssh alias must resolve.
#
# Two things here look like bugs and are not:
#   * anton's vLLM answers on 172.17.0.1:8000 but times out on 127.0.0.1:8000.
#     A host firewall rule drops loopback to that port; connections hang rather
#     than being refused. VLLM_BASE_URL points at the reachable address.
#   * Every authenticated endpoint reports 401 when healthy. 401 means the
#     process is up and rejecting an unauthenticated probe, which is the point.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ANTON_UNITS=(vllm-qwen membraneclaw-agent)
TEMUR_UNITS=(reaktoro-mcp)

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
fail=0

say()  { printf '\n%s== %s ==%s\n' "$DIM" "$1" "$OFF"; }
mark() { # mark <label> <actual> <expected...>
  local label=$1 got=$2; shift 2
  for want in "$@"; do
    if [[ $got == "$want" ]]; then printf '  %s%-38s %s%s\n' "$GRN" "$label" "$got" "$OFF"; return 0; fi
  done
  printf '  %s%-38s %s (want: %s)%s\n' "$RED" "$label" "$got" "$*" "$OFF"; fail=1; return 1
}
code() { curl -s -o /dev/null -m "${2:-8}" -w '%{http_code}' "$1" 2>/dev/null; }
post() { curl -s -o /dev/null -m "${2:-8}" -w '%{http_code}' -X POST "$1" \
           -H 'content-type: application/json' -d '{}' 2>/dev/null; }

status_units() {
  say "units"
  for u in "${ANTON_UNITS[@]}"; do mark "anton/$u" "$(systemctl is-active "$u" 2>&1)" active; done
  printf '  %s%-38s %s%s\n' "$DIM" "anton/open-webui (docker)" \
    "$(docker ps --filter name=open-webui --format '{{.Status}}' 2>/dev/null || echo '?')" "$OFF"
  for u in "${TEMUR_UNITS[@]}"; do
    mark "temur/$u" "$(ssh temur "systemctl is-active $u" 2>&1)" active
  done
}

status_endpoints() {
  say "endpoints"
  # 401 = up and gating correctly. See header note on 172.17.0.1.
  mark "anton vLLM   172.17.0.1:8000"  "$(code http://172.17.0.1:8000/v1/models)" 401 200
  mark "anton agent  127.0.0.1:8001"   "$(code http://127.0.0.1:8001/v1/models)" 200 401
  mark "anton webui  127.0.0.1:3000"   "$(code http://127.0.0.1:3000/health)"    200
  mark "temur mcp    127.0.0.1:8003"   "$(ssh temur 'curl -s -o /dev/null -m 8 -w "%{http_code}" -X POST http://127.0.0.1:8003/mcp -H "content-type: application/json" -d "{}"' 2>&1)" 401

  say "public (Tailscale Funnel)"
  mark "anton  /            -> webui"  "$(code https://anton.tail35bed8.ts.net/ 12)" 200
  mark "temur  /mcp         -> mcp"    "$(post https://temur.tail35bed8.ts.net/mcp 12)" 401
  mark "temur  /.well-known/oauth-authorization-server" \
       "$(code https://temur.tail35bed8.ts.net/.well-known/oauth-authorization-server 12)" 200
}

status_chat() {
  say "end-to-end (Open WebUI -> agent -> vLLM)"
  local key
  # AGENT_API_KEY specifically: the agent gates on its own key, not vLLM's, and
  # the two are different values.
  key=$(grep -E '^AGENT_API_KEY=' .env 2>/dev/null | cut -d= -f2-)
  if [[ -z $key ]]; then
    printf '  %s%-38s no API key in .env, skipped%s\n' "$YEL" "chat round-trip" "$OFF"; return
  fi
  local out
  out=$(curl -s -m 120 http://127.0.0.1:8001/v1/chat/completions \
        -H 'content-type: application/json' -H "Authorization: Bearer $key" \
        -d '{"model":"qwen3.5-9b-agent","messages":[{"role":"user","content":"reply with the single word: alive"}],"stream":false}' \
        | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:40] or "(empty)")
except Exception as e: print("BAD:", e)' 2>&1)
  if [[ $out == BAD:* || -z $out ]]; then
    printf '  %s%-38s %s%s\n' "$RED" "chat round-trip" "$out" "$OFF"; fail=1
  else
    printf '  %s%-38s %s%s\n' "$GRN" "chat round-trip" "$out" "$OFF"
  fi
}

case "${1:-status}" in
  up)
    say "starting"
    for u in "${ANTON_UNITS[@]}"; do
      [[ $(systemctl is-active "$u") == active ]] || { echo "  anton: starting $u"; sudo systemctl start "$u"; }
    done
    docker ps --filter name=open-webui --format '{{.Names}}' | grep -q . \
      || { echo "  anton: starting open-webui"; docker start open-webui >/dev/null; }
    for u in "${TEMUR_UNITS[@]}"; do
      [[ $(ssh temur "systemctl is-active $u") == active ]] || { echo "  temur: starting $u"; ssh temur "sudo systemctl start $u"; }
    done
    echo "  waiting for vLLM to load weights (~60s on a cold start)"
    for _ in $(seq 1 30); do [[ $(code http://172.17.0.1:8000/v1/models 3) != 000 ]] && break; sleep 3; done
    status_units; status_endpoints; status_chat
    ;;
  logs)
    for u in "${ANTON_UNITS[@]}"; do
      say "anton/$u"; journalctl -u "$u" --no-pager -p warning -n 15 2>/dev/null | tail -15
    done
    for u in "${TEMUR_UNITS[@]}"; do
      say "temur/$u"; ssh temur "journalctl -u $u --no-pager -p warning -n 15" 2>/dev/null | tail -15
    done
    ;;
  status)
    status_units; status_endpoints; status_chat
    ;;
  *)
    echo "usage: $0 [status|up|logs]" >&2; exit 2
    ;;
esac

say "result"
if (( fail )); then echo "  ${RED}something is down — see red lines above${OFF}"; else echo "  ${GRN}all green${OFF}"; fi
exit $fail
