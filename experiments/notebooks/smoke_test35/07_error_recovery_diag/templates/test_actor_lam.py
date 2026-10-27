"""CPU check of --actor-lam (gae_turn's actor_lam: the actor's lambda on every turn, the critic's targets from lam).
  1. gae_turn without actor_lam is bit-identical to 86d8cce's (the commit before the 07 lambda flags); actor_lam=lam is
     bit-identical to the default; for actor_lam 0 and 0.5 the returns are bit-identical to the default and every
     turn's advantage equals the closed form sum_k (gamma actor_lam)^(k-j) delta_k, computed independently here.
  2. With first_lam as well: the first turn is delta_0 + first_lam * A_1 (A_1 from actor_lam), the other turns as above.
  3. A hand-computed three-turn episode; bad values refused."""
import importlib.util, subprocess, sys, types
import torch

HERE = "/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35"
spec = importlib.util.spec_from_file_location("nb07", f"{HERE}/07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
head_src = subprocess.run(["git", "-C", HERE, "show", "86d8cce:experiments/notebooks/smoke_test35/07_error_recovery.py"],
                          capture_output=True, text=True, check=True).stdout
old = types.ModuleType("nb07_head"); old.__file__ = f"{HERE}/07_error_recovery.py"
exec(compile(head_src, "nb07_head", "exec"), old.__dict__)


def batch(seed, N=8, L=40):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, L + 1, (N,), generator=g)
    mask = (torch.arange(L)[None] < lens[:, None]).float()          # compacted policy tokens, as build_batch
    turn = torch.zeros(N, L, dtype=torch.long)
    for i in range(N):
        cuts = sorted(torch.randint(1, max(2, int(lens[i])), (int(torch.randint(0, 5, (1,), generator=g)),), generator=g).tolist())
        for c in cuts:
            turn[i, c:] += 1
    rewards = torch.randn(N, L, generator=g) * 0.1 * mask
    values = torch.randn(N, L, generator=g) * 0.3 * mask
    return rewards, values, mask, turn


def by_hand(r, v, m, t, gamma, lam, first_lam=None):
    """Per episode and turn, the closed-form advantage: turn rewards and turn-start values, deltas, discounted sums."""
    out = torch.zeros_like(v)
    for i in range(m.shape[0]):
        n = int(m[i].sum())
        if not n:
            continue
        tt = t[i, :n]
        starts = [k for k in range(n) if k == 0 or tt[k] != tt[k - 1]]
        ends = starts[1:] + [n]
        R = [float(r[i, a:b].sum()) for a, b in zip(starts, ends)]
        V = [float(v[i, a]) for a in starts] + [0.0]
        d = [R[j] + gamma * V[j + 1] - V[j] for j in range(len(R))]
        A = [sum((gamma * lam) ** (k - j) * d[k] for k in range(j, len(d))) for j in range(len(d))]
        if first_lam is not None and len(A) > 1:
            A[0] = d[0] + gamma * first_lam * A[1]
        for (a, b), x in zip(zip(starts, ends), A):
            out[i, a:b] = x
    return out


worst = {"default vs 86d8cce (must be 0)": 0.0, "actor_lam=lam vs default (must be 0)": 0.0,
         "actor_lam 0 / 0.5: returns vs default (must be 0)": 0.0, "actor_lam 0 / 0.5: advantages vs closed form": 0.0,
         "actor_lam 0.5 + first_lam 0 / 0.5 vs closed form": 0.0, "default vs closed form (lam 0.95)": 0.0}
for s in range(300):
    r, v, m, t = batch(s)
    a_new, R_new = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95)
    a_old, R_old = old.gae_turn(r, v, m, t, gamma=1.0, lam=0.95)
    up = lambda k, x: worst.__setitem__(k, max(worst[k], float(x)))
    up("default vs 86d8cce (must be 0)", max((a_new - a_old).abs().max(), (R_new - R_old).abs().max()))
    up("default vs closed form (lam 0.95)", (a_new - by_hand(r, v, m, t, 1.0, 0.95)).abs().max())
    a_l, R_l = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, actor_lam=0.95)
    up("actor_lam=lam vs default (must be 0)", max((a_l - a_new).abs().max(), (R_l - R_new).abs().max()))
    for al in (0.0, 0.5):
        a, R = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, actor_lam=al)
        up("actor_lam 0 / 0.5: returns vs default (must be 0)", (R - R_new).abs().max())
        up("actor_lam 0 / 0.5: advantages vs closed form", (a - by_hand(r, v, m, t, 1.0, al)).abs().max())
    for fl in (0.0, 0.5):
        a, R = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, actor_lam=0.5, first_lam=fl)
        up("actor_lam 0 / 0.5: returns vs default (must be 0)", (R - R_new).abs().max())
        up("actor_lam 0.5 + first_lam 0 / 0.5 vs closed form", (a - by_hand(r, v, m, t, 1.0, 0.5, first_lam=fl)).abs().max())
for k, x in worst.items():
    print(f"  {k:52} {x:.2e}")
assert worst["default vs 86d8cce (must be 0)"] == 0 and worst["actor_lam=lam vs default (must be 0)"] == 0
assert worst["actor_lam 0 / 0.5: returns vs default (must be 0)"] == 0
assert max(worst[k] for k in ("actor_lam 0 / 0.5: advantages vs closed form", "actor_lam 0.5 + first_lam 0 / 0.5 vs closed form",
                              "default vs closed form (lam 0.95)")) < 1e-6

# hand-computed: three turns, rewards per turn [0.0, 0.25, 0.5], values at turn starts [0.4, 0.3, 0.6]
m = torch.ones(1, 6); t = torch.tensor([[0, 0, 1, 1, 2, 2]])
r = torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.0, 0.5]]); v = torch.tensor([[0.4, 9, 0.3, 9, 0.6, 9]])
d = [0.0 + 0.3 - 0.4, 0.25 + 0.6 - 0.3, 0.5 - 0.6]                  # -0.1, +0.55, -0.1
R95 = [0.4 + d[0] + 0.95 * d[1] + 0.95 ** 2 * d[2], 0.3 + d[1] + 0.95 * d[2], 0.6 + d[2]]   # the critic's targets at lam 0.95
for al, want in ((0.95, [d[0] + 0.95 * d[1] + 0.95 ** 2 * d[2], d[1] + 0.95 * d[2], d[2]]),
                 (0.5, [d[0] + 0.5 * d[1] + 0.25 * d[2], d[1] + 0.5 * d[2], d[2]]),
                 (0.0, d)):
    a, R = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, actor_lam=al)
    got, gotR = [float(a[0, k]) for k in (0, 2, 4)], [float(R[0, k]) for k in (0, 2, 4)]
    assert all(abs(x - y) < 1e-6 for x, y in zip(got, want)), (al, got, want)
    assert all(abs(x - y) < 1e-6 for x, y in zip(gotR, R95)), (al, gotR, R95)
    print(f"  three turns, actor_lam {al}: advantages {[round(x, 4) for x in got]} (by hand {[round(x, 4) for x in want]}), "
          f"returns {[round(x, 4) for x in gotR]}")

for bad in ({"actor_lam": 1.5, "adv_unit": "turn"}, {"actor_lam": 0.5, "adv_unit": "token"}):
    try:
        nb.train_ppo({**nb.PPO_DEFAULTS, **bad}, None)
        raise AssertionError(f"accepted {bad}")
    except SystemExit as e:
        print("  bad value:", bad, "->", e)
print("ok")
