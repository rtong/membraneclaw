import json, sys
from collections import defaultdict
for run in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f"ra_{run}.jsonl")]
    meta=[r for r in rows if r.get("_meta")]; eps=[r for r in rows if not r.get("_meta")]
    nstep=len({r["step"] for r in eps})
    J1=meta[-1].get("J_225"); J0=meta[0]["J_200"]
    print(f"== {run}: {len(eps)} episodes, {nstep} steps; actual dJ multi {J1['J multi']-J0['J multi']:+.2f}, one {J1['J one unit']-J0['J one unit']:+.2f}, L0 {J1['J dev L0']-J0['J dev L0']:+.2f}")
    for target in ("J multi","J one unit"):
        acc=defaultdict(float); n=defaultdict(int); adv=defaultdict(float)
        for r in eps:
            if "dJ" not in r: continue
            k=(str(r["group"]), str(r["first"]))
            n[k]+=1; adv[k]+=r["adv_task"]
            for p,v in r["dJ"][target].items(): acc[k+(p,)]+=v
        print(f"  {target}: per-step push, by episode group x first call   (task / kl / centre ; n, mean task adv)")
        tot=defaultdict(float)
        for k in sorted(n):
            t,kl,c=(acc[k+(p,)]/nstep for p in ("task","kl","centre"))
            tot["task"]+=t; tot["kl"]+=kl; tot["centre"]+=c
            if abs(t)+abs(kl)+abs(c) > 0.005:
                print(f"    {k[0]:12} {k[1]:9} task {t:+.3f} kl {kl:+.3f} centre {c:+.3f}   n {n[k]:3} adv {adv[k]/n[k]:+.3f}")
        print(f"    total task {tot['task']:+.3f} kl {tot['kl']:+.3f} centre {tot['centre']:+.3f}")
    v=sorted((r["dJ"]["J multi"]["task"] for r in eps if "dJ" in r), key=abs, reverse=True)
    s=sum(v); print(f"  task push on J multi: sum {s/nstep:+.3f}/step; top-5 episodes {sum(v[:5])/nstep:+.3f}, top-20 {sum(v[:20])/nstep:+.3f}, rest {sum(v[20:])/nstep:+.3f}")
    top=sorted((r for r in eps if "dJ" in r), key=lambda r: abs(r["dJ"]["J multi"]["task"]), reverse=True)[:8]
    for r in top: print(f"     step {r['step']} {str(r['group']):12} {str(r['first']):9} reward {r['reward']:.2f} adv {r['adv_task']:+.3f} p(route) {2.718281828**r['lp_route']:.3f} T {r['T']:.0f} push {r['dJ']['J multi']['task']/nstep:+.3f}")
