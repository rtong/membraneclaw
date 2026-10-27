#!/usr/bin/env bash
# 08.5: wait until the run writes a new evaluation (or its run script ends), then print a short summary.
#   08.5_mlx_stack_tools/watch_eval.sh <run dir> <run script pid>
D=$1; P=$2
F=$D/eval.jsonl
n=$( [ -f "$F" ] && wc -l < "$F" || echo 0 )
while kill -0 "$P" 2>/dev/null; do
    m=$( [ -f "$F" ] && wc -l < "$F" || echo 0 )
    [ "$m" -gt "$n" ] && break
    sleep 60
done
date '+%m-%d %H:%M'
kill -0 "$P" 2>/dev/null && echo "run script alive" || echo "RUN SCRIPT ENDED"
[ -f "$D/ending.json" ] && { echo "ending:"; cat "$D/ending.json"; }
python3 - "$D" <<'EOF'
import json, sys
d = sys.argv[1]
ev = [json.loads(l) for l in open(f"{d}/eval.jsonl") if l.strip()]
print("evals:", [(r["step"], round(r["reward"], 4)) for r in ev])
r = ev[-1]
print(f"last eval @{r['step']}: reward {r['reward']:.4f} decision {r['decision_correct']:.3f} evidence {r['evidence']:.3f} "
      f"erred {r['recovery']['sim_erred']:.3f} recovered {r['recovery']['recovered_given_error']}")
m = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
if m:
    act = [x for x in m if not x.get("critic_warmup")]
    last = m[-1]
    print(f"last step {last['step']}: reward {last['reward_mean']:.4f} step_seconds {last.get('step_seconds', 0):.0f} "
          f"gen {last.get('gen_seconds', 0):.0f}; actor steps {len(act)}"
          + (f", mean step_seconds of the last 10 actor steps {sum(x['step_seconds'] for x in act[-10:]) / len(act[-10:]):.0f}" if act else ""))
EOF
pmset -g batt | head -2 | tail -1
