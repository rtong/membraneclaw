"""CPU check of --first-turn-lam (gae_turn's first_lam) and --critic-chunk-tokens (critic_fit_chunked).
  1. gae_turn without first_lam is bit-identical to 86d8cce's (the commit before these flags); first_lam=0 gives the first turn
     r_0 + V(s_1) - V(s_0) and leaves the other turns and every return unchanged; first_lam=lam equals the default;
     a hand-computed three-turn episode.
  2. critic_fit_chunked: the same head after 8 epochs as the whole-window fit (value_loss), any chunk size."""
import copy, importlib.util, subprocess, sys, types
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
        cuts = sorted(torch.randint(1, max(2, int(lens[i])), (int(torch.randint(0, 4, (1,), generator=g)),), generator=g).tolist())
        for c in cuts:
            turn[i, c:] += 1
    rewards = torch.randn(N, L, generator=g) * 0.1 * mask
    values = torch.randn(N, L, generator=g) * 0.3 * mask
    return rewards, values, mask, turn


worst = {"default vs 86d8cce (must be 0)": 0.0, "first_lam=lam vs default": 0.0, "first_lam=0: returns vs default (must be 0)": 0.0,
         "first_lam=0: other turns vs default (must be 0)": 0.0, "first_lam=0: first turn vs r0+V1-V0": 0.0}
for s in range(300):
    r, v, m, t = batch(s)
    a_new, R_new = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95)
    a_old, R_old = old.gae_turn(r, v, m, t, gamma=1.0, lam=0.95)
    worst["default vs 86d8cce (must be 0)"] = max(worst["default vs 86d8cce (must be 0)"], float((a_new - a_old).abs().max()), float((R_new - R_old).abs().max()))
    a_l, _ = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, first_lam=0.95)
    worst["first_lam=lam vs default"] = max(worst["first_lam=lam vs default"], float((a_l - a_new).abs().max()))
    a0, R0 = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, first_lam=0.0)
    worst["first_lam=0: returns vs default (must be 0)"] = max(worst["first_lam=0: returns vs default (must be 0)"], float((R0 - R_new).abs().max()))
    for i in range(m.shape[0]):
        n = int(m[i].sum())
        tt = t[i, :n]
        first = tt == tt[0]
        worst["first_lam=0: other turns vs default (must be 0)"] = max(worst["first_lam=0: other turns vs default (must be 0)"],
                                                                     float((a0[i, :n][~first] - a_new[i, :n][~first]).abs().max()) if (~first).any() else 0.0)
        if (~first).any():
            k1 = int(torch.nonzero(~first)[0])                     # the second turn's first token
            want = float(r[i, :n][first].sum() + v[i, k1] - v[i, 0])
            worst["first_lam=0: first turn vs r0+V1-V0"] = max(worst["first_lam=0: first turn vs r0+V1-V0"], float((a0[i, :n][first] - want).abs().max()))
for k, x in worst.items():
    print(f"  {k:48} {x:.2e}")
assert worst["default vs 86d8cce (must be 0)"] == 0 and worst["first_lam=0: returns vs default (must be 0)"] == 0
assert worst["first_lam=0: other turns vs default (must be 0)"] == 0 and worst["first_lam=0: first turn vs r0+V1-V0"] < 1e-6

# hand-computed: three turns, rewards per turn [0.0, 0.25, 0.5], values at turn starts [0.4, 0.3, 0.6]
m = torch.ones(1, 6); t = torch.tensor([[0, 0, 1, 1, 2, 2]])
r = torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.0, 0.5]]); v = torch.tensor([[0.4, 9, 0.3, 9, 0.6, 9]])
d = [0.0 + 0.3 - 0.4, 0.25 + 0.6 - 0.3, 0.5 - 0.6]
A2 = d[2]; A1 = d[1] + 0.95 * A2; A0 = d[0] + 0.95 * A1
for fl, want0 in ((None, A0), (0.0, d[0]), (0.5, d[0] + 0.5 * A1)):
    a, R = nb.gae_turn(r, v, m, t, gamma=1.0, lam=0.95, first_lam=fl)
    got = [float(a[0, k]) for k in (0, 2, 4)]
    assert all(abs(x - y) < 1e-6 for x, y in zip(got, [want0, A1, A2])), (fl, got)
    print(f"  three turns, first_lam {fl}: advantages {[round(x, 4) for x in got]} (first turn by hand {want0:+.4f})")

# 2. the chunked critic fit
torch.manual_seed(0)
window = [(torch.randn(300 + 37 * k, 16), torch.randn(300 + 37 * k) * 0.3, torch.rand(300 + 37 * k)) for k in range(5)]
h0 = nb._value_head(16, 0.4)
torch.nn.init.normal_(h0.weight, std=0.1)
for chunk in (64, 500, 10**6):
    ha, hb = copy.deepcopy(h0), copy.deepcopy(h0)
    oa, ob = (torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=0.0) for h in (ha, hb))
    fh, fv, fr = (torch.cat([w[k] for w in window]) for k in range(3))
    for _ in range(8):
        oa.zero_grad(set_to_none=True)
        vl, sa = nb.value_loss(nb._values(ha, fh), fv, fr, clip_eps=0.2)
        vl.backward(); torch.nn.utils.clip_grad_norm_(ha.parameters(), 1.0); oa.step()
    sb = nb.critic_fit_chunked(hb, ob, window, epochs=8, clip_eps=0.2, chunk=chunk, device="cpu")
    dw = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(ha.parameters(), hb.parameters()))
    ds = max(abs(sa[k] - sb[k]) for k in sa)
    print(f"  critic chunk {chunk:>7}: max relative parameter difference {dw:.2e}, stats difference {ds:.2e}")
    assert dw < 1e-5 and ds < 1e-5

for bad in ({"first_turn_lam": 1.5, "adv_unit": "turn"}, {"first_turn_lam": 0.0, "adv_unit": "token"}):
    try:
        nb.train_ppo({**nb.PPO_DEFAULTS, **bad}, None)
    except SystemExit as e:
        print("  bad value:", bad, "->", e)
