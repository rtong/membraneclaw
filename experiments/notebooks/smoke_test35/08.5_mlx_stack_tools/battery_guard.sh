#!/usr/bin/env bash
# 08.5: pause (SIGSTOP) smoke_test35's training / evaluation python while the battery is low, resume (SIGCONT) once it
# has recharged. Only the main process is stopped: the WaterTAP workers finish their call and wait on the pipe, and the
# main process has no wall-clock timeouts on tool calls. The step watchdog (STEP_WATCHDOG) counts the pause.
# Killing the guard resumes anything it paused -- after its current `sleep 30` (bash runs the trap then).
#   LOW=5 HIGH=50 nohup 08.5_mlx_stack_tools/battery_guard.sh > /dev/null 2>&1 &        log: /tmp/battery_guard.log
LOW=${LOW:-5}
HIGH=${HIGH:-20}
LOG=${LOG:-/tmp/battery_guard.log}
PAT='membrane_grpo/.venv/bin/python 0[0-9][._]'
paused=""
resume_all() {
    for p in $paused; do kill -CONT "$p" 2>/dev/null; done
    paused=""
}
trap 'resume_all; echo "$(date "+%F %T") guard exit (resumed anything paused)" >> "$LOG"; trap - EXIT; exit 0' EXIT INT TERM HUP
echo "$(date '+%F %T') guard start: pause at <= $LOW%, resume at >= $HIGH%" >> "$LOG"
i=0
while true; do
    batt=$(pmset -g batt)
    pct=$(echo "$batt" | grep -Eo '[0-9]+%' | head -1 | tr -d %)
    src=$(echo "$batt" | head -1 | grep -o "'.*'")
    if [ -n "$pct" ]; then
        if [ -z "$paused" ] && [ "$pct" -le "$LOW" ]; then
            for p in $(pgrep -f "$PAT"); do kill -STOP "$p" && paused="$paused $p"; done
            [ -n "$paused" ] && echo "$(date '+%F %T') battery $pct% $src <= $LOW%: paused$paused" >> "$LOG"
        elif [ -n "$paused" ] && [ "$pct" -ge "$HIGH" ]; then
            echo "$(date '+%F %T') battery $pct% $src >= $HIGH%: resumed$paused" >> "$LOG"
            resume_all
        fi
        [ $((i % 20)) -eq 0 ] && echo "$(date '+%F %T') battery $pct% $src${paused:+ (paused)}" >> "$LOG"
    fi
    i=$((i + 1))
    sleep 30
done
