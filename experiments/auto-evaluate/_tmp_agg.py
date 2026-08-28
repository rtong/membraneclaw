import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")

RUNS = ["formal-anchor-20260820", "v069-rep1"]


def load_run(run):
    run_dir = os.path.join(r"f:\MembraneClaw\ScrapingPipe\auto-evaluate\runs", run)
    with open(os.path.join(run_dir, "judge_mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]
    m = {x["task_id"]: x["system_id"] for x in mapping}
    ratings = []
    with open(os.path.join(run_dir, "ratings.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ratings.append(json.loads(line))
    resp = {}
    resp_dir = os.path.join(run_dir, "responses")
    if os.path.isdir(resp_dir):
        for fn in os.listdir(resp_dir):
            if fn.endswith(".json"):
                rec = json.load(open(os.path.join(resp_dir, fn), encoding="utf-8"))
                resp[(rec["case_id"], rec["system_id"])] = rec
    return ratings, m, resp


for run in RUNS:
    ratings, m, resp = load_run(run)
    print(f"===== {run}: ratings parsed: {len(ratings)}")
    by_sys = {}
    for r in ratings:
        sys_id = m.get(r["task_id"], "?")
        by_sys.setdefault(sys_id, []).append(r)

    for sys_id, rows in sorted(by_sys.items()):
        if str(sys_id).startswith("gpt-5.6-teacher"):
            mean = sum(r["total_score"] for r in rows) / len(rows)
            print(f"SYS={sys_id:20s} n={len(rows)} mean={mean:.1f} (reference)")
            continue
        ok_rows = [
            r for r in rows
            if resp.get((r["case_id"], sys_id), {}).get("status") == "success"
        ]
        err_rows = [
            r for r in rows
            if resp.get((r["case_id"], sys_id), {}).get("status") != "success"
        ]
        ctx = sum(
            1 for r in err_rows
            if (resp.get((r["case_id"], sys_id), {}).get("error_type") or "") == "context_window_exceeded"
        )
        mean_all = sum(r["total_score"] for r in rows) / len(rows)
        mean_ok = sum(r["total_score"] for r in ok_rows) / len(ok_rows) if ok_rows else float("nan")
        mean_err = sum(r["total_score"] for r in err_rows) / len(err_rows) if err_rows else float("nan")
        print(
            f"SYS={sys_id:20s} n={len(rows)} mean={mean_all:.1f} "
            f"mean_ok={mean_ok:.1f}(n={len(ok_rows)}) mean_err={mean_err:.1f}(n={len(err_rows)}) "
            f"ctx_overflow={ctx}"
        )

    rag_events = 0
    for rec in resp.values():
        raw = rec.get("raw_response") or {}
        if raw.get("trajectory_events"):
            rag_events += 1
        summ = (rec.get("trajectory") or {}).get("summary") or {}
        if summ.get("retrieval_interactions"):
            rag_events += 1
    print(f"responses with observable retrieval events: {rag_events}/{len(resp)}")

    by_case_sys = {}
    for r in ratings:
        sys_id = m.get(r["task_id"], "?")
        by_case_sys.setdefault(r["case_id"], {})[sys_id] = r["total_score"]
    sysids = sorted({m.get(r["task_id"]) for r in ratings})
    print(f"case | " + " | ".join(sysids))
    for case in sorted(by_case_sys):
        row = by_case_sys[case]
        print(f"{case:65s} | " + " | ".join(f"{row.get(s, float('nan')):.1f}" for s in sysids))
    print()
