"""CPU check of --loss-norm: the actor loop's two branches, as train_ppo runs them, on random batches.
  episode: the per-episode token mean averaged over episodes (the statement is unchanged; checked against the
           closed form, and bit-for-bit against the same loop run from 86d8cce, the commit before these flags).
  batch:   equals one PPO loss averaged over all the batch's trainable tokens (closed form)."""
import importlib.util, subprocess, sys, types
import torch

HERE = "/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35"
spec = importlib.util.spec_from_file_location("nb07", f"{HERE}/07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
head_src = subprocess.run(["git", "-C", HERE, "show", "86d8cce:experiments/notebooks/smoke_test35/07_error_recovery.py"],
                          capture_output=True, text=True, check=True).stdout
old = types.ModuleType("nb07_head"); old.__file__ = f"{HERE}/07_error_recovery.py"; exec(compile(head_src, "nb07_head", "exec"), old.__dict__)


def loop(per_batch, theta, old_lp, adv, actor_mask, loss_fn):
    """train_ppo's inner actor epoch, lines as in the file (forward_policy replaced by the leaf log-probs)."""
    theta.grad = None
    trainable = actor_mask.sum(dim=-1)
    batch_tokens = trainable.sum().clamp(min=1.0)
    for i in range(len(theta)):
        lp = theta[i:i + 1]
        loss, st_ = loss_fn(lp, old_lp[i:i + 1], adv[i:i + 1], actor_mask[i:i + 1], clip_eps=0.2)
        if per_batch:
            (loss * (trainable[i] / batch_tokens)).backward()
        else:
            (loss / len(theta)).backward()
    return theta.grad.clone()


def closed(theta, old_lp, adv, m, per_batch):
    theta = theta.detach().clone().requires_grad_(True)
    r = torch.exp(theta - old_lp)
    obj = torch.min(r * adv, torch.clamp(r, 0.8, 1.2) * adv) * m
    if per_batch:
        loss = -obj.sum() / m.sum()
    else:
        t = m.sum(-1)
        loss = -(obj.sum(-1) / t.clamp(min=1.0)).sum() / len(t)
    loss.backward()
    return theta.grad


worst = {"episode": 0.0, "batch": 0.0, "batch vs episode, equal lengths": 0.0}
for trial in range(200):
    g = torch.Generator().manual_seed(trial)
    N, L = 8, 64
    lens = torch.randint(0, L + 1, (N,), generator=g)
    if trial % 3 == 0:
        lens[trial % N] = 0                                  # an episode with no trainable token
    m = (torch.arange(L)[None] >= L - lens[:, None]).float()  # left-padded, as build_batch
    m = m * (torch.rand(N, L, generator=g) > 0.15).float()     # frozen turn ends / answer scaffolding
    old_lp = -torch.rand(N, L, generator=g) * 3
    theta = (old_lp + torch.randn(N, L, generator=g) * 0.3).requires_grad_(True)   # ratios well past the clip
    adv = torch.randn(N, L, generator=g) * m
    for name, pb in (("episode", False), ("batch", True)):
        got = loop(pb, theta, old_lp, adv, m, nb.ppo_actor_loss)
        ref = closed(theta, old_lp, adv, m, pb)
        worst[name] = max(worst[name], float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-12)))
    # the episode branch bit-for-bit against 86d8cce's own ppo_actor_loss and statement
    assert torch.equal(loop(False, theta, old_lp, adv, m, nb.ppo_actor_loss), loop(False, theta, old_lp, adv, m, old.ppo_actor_loss))
    # equal trainable counts: the two norms are the same up to rounding
    m2 = torch.ones(N, L)
    worst["batch vs episode, equal lengths"] = max(worst["batch vs episode, equal lengths"], float(
        (loop(True, theta, old_lp, adv, m2, nb.ppo_actor_loss) - loop(False, theta, old_lp, adv, m2, nb.ppo_actor_loss)).abs().max()))
print("max relative gradient error vs closed form (200 random batches):")
for k, v in worst.items():
    print(f"  {k:34} {v:.2e}")

# the flag: the parser makes --loss-norm from PPO_DEFAULTS; a bad value stops train_ppo before anything loads
assert nb.PPO_DEFAULTS["loss_norm"] == "episode" and "loss_norm" not in old.PPO_DEFAULTS
try:
    nb.train_ppo({**nb.PPO_DEFAULTS, "loss_norm": "token"}, None)
except SystemExit as e:
    print("bad value:", e)
