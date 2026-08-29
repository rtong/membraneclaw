"""Tests for the GRPO update, checked against the gradient derived by hand.

`toy_mdp/ppo.py` writes the clipped surrogate's derivative out by hand and this
file is the same check one level up: autograd carries the chain rule through the
network, but the derivative with respect to the log-probabilities is still
something we can state in closed form and verify.

With `rho = exp(logp - logp_old)`:

    d/d logp  min(rho*A, clip(rho)*A)  =  rho * A,  unless the clip binds
                                          0,        when A > 0 and rho > 1+eps
                                          0,        when A < 0 and rho < 1-eps

Everything here runs on CPU with tiny tensors -- no model, no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from grpo_scratch import (  # noqa: E402
    build_mask,
    completion_logprobs,
    grpo_surrogate,
    selective_logprobs,
)

EPS = 0.2


def closed_form_grad(logprobs, old_logprobs, advantages, mask, clip_eps=EPS):
    """The hand-derived gradient of the loss with respect to `logprobs`."""
    ratio = torch.exp(logprobs - old_logprobs)
    advantage = advantages.unsqueeze(-1)
    binding = ((advantage > 0) & (ratio > 1 + clip_eps)) | (
        (advantage < 0) & (ratio < 1 - clip_eps)
    )
    grad = torch.where(binding, torch.zeros_like(ratio), ratio * advantage)
    # loss = -sum(objective * mask) / sum(mask)
    return -grad * mask / mask.sum()


def sample(batch=3, tokens=5, seed=0, spread=0.3):
    torch.manual_seed(seed)
    old = torch.randn(batch, tokens)
    logprobs = (old + spread * torch.randn(batch, tokens)).requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 0.5])[:batch]
    mask = torch.ones(batch, tokens)
    return logprobs, old, advantages, mask


# --- the gradient ---------------------------------------------------------------


def test_autograd_matches_the_hand_derived_gradient():
    logprobs, old, advantages, mask = sample(spread=0.05)
    loss, _ = grpo_surrogate(logprobs, old, advantages, mask, clip_eps=EPS)
    loss.backward()

    expected = closed_form_grad(logprobs.detach(), old, advantages, mask)
    assert torch.allclose(logprobs.grad, expected, atol=1e-6)


def test_the_derivation_still_holds_once_the_clip_binds():
    """A wide spread pushes ratios past the band in both directions."""
    logprobs, old, advantages, mask = sample(spread=1.5, seed=7)
    ratio = torch.exp(logprobs.detach() - old)
    assert (ratio > 1 + EPS).any() and (ratio < 1 - EPS).any(), "clip never engaged"

    loss, stats = grpo_surrogate(logprobs, old, advantages, mask, clip_eps=EPS)
    loss.backward()

    expected = closed_form_grad(logprobs.detach(), old, advantages, mask)
    assert torch.allclose(logprobs.grad, expected, atol=1e-6)
    assert stats["clip_frac"] > 0.0


def test_at_ratio_one_the_surrogate_is_the_vanilla_policy_gradient():
    """One inner epoch means rho == 1 exactly, so the clip is inert.

    GRPO with a single update per batch *is* REINFORCE with a group baseline;
    the clip only starts working when the batch is reused.
    """
    logprobs, _, advantages, mask = sample()
    old = logprobs.detach().clone()

    loss, stats = grpo_surrogate(logprobs, old, advantages, mask, clip_eps=EPS)
    loss.backward()

    expected = -advantages.unsqueeze(-1).expand_as(mask) * mask / mask.sum()
    assert torch.allclose(logprobs.grad, expected, atol=1e-6)
    assert stats["ratio_mean"] == pytest.approx(1.0)
    assert stats["clip_frac"] == 0.0


def test_a_clipped_token_contributes_nothing():
    logprobs = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])  # rho = e ~ 2.72, far above 1+eps
    loss, _ = grpo_surrogate(
        logprobs, old, torch.tensor([1.0]), torch.ones(1, 1), clip_eps=EPS
    )
    loss.backward()
    assert logprobs.grad.abs().max() == 0.0


def test_a_negative_advantage_clips_on_the_other_side():
    logprobs = torch.tensor([[-1.0]], requires_grad=True)
    old = torch.tensor([[0.0]])  # rho ~ 0.37, far below 1-eps
    loss, _ = grpo_surrogate(
        logprobs, old, torch.tensor([-1.0]), torch.ones(1, 1), clip_eps=EPS
    )
    loss.backward()
    assert logprobs.grad.abs().max() == 0.0


def test_a_zero_advantage_group_produces_no_gradient():
    """The degenerate case the run counts as `adv_zero_frac`."""
    logprobs, old, _, mask = sample()
    loss, _ = grpo_surrogate(logprobs, old, torch.zeros(3), mask)
    loss.backward()
    assert logprobs.grad.abs().max() == 0.0


# --- masking --------------------------------------------------------------------


def test_padding_is_never_scored():
    logprobs, old, advantages, mask = sample()
    mask[:, 3:] = 0.0
    loss, _ = grpo_surrogate(logprobs, old, advantages, mask)
    loss.backward()
    assert logprobs.grad[:, 3:].abs().max() == 0.0
    assert logprobs.grad[:, :3].abs().max() > 0.0


def test_eos_is_scored_but_padding_never_is():
    """Deciding to stop is an action; padding was never sampled."""
    ids = torch.tensor([[5, 6, 2, 9, 9], [7, 8, 9, 9, 2]])
    mask = build_mask(ids, eos_id=2, pad_id=9)
    assert mask[0].tolist() == [1, 1, 1, 0, 0], "EOS scored, everything after is not"
    assert mask[1].tolist() == [1, 1, 0, 0, 0], "the first pad is not scored"


def test_an_aliased_pad_token_still_scores_the_eos():
    """Qwen2.5 has no distinct pad token, so load_policy aliases it to EOS."""
    ids = torch.tensor([[5, 6, 2, 2, 2]])
    assert build_mask(ids, eos_id=2, pad_id=2)[0].tolist() == [1, 1, 1, 0, 0]


def test_a_completion_that_never_stops_is_scored_throughout():
    ids = torch.tensor([[5, 6, 7, 8]])
    assert build_mask(ids, eos_id=2, pad_id=9)[0].tolist() == [1, 1, 1, 1]


# --- normalisation ---------------------------------------------------------------


def test_sequence_normalisation_weights_short_completions_more_per_token():
    """Why `token` is the default: length is a metric here, not a nuisance."""
    logprobs = torch.zeros(2, 6, requires_grad=True)
    old = torch.zeros(2, 6)
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[1.0] * 2 + [0.0] * 4, [1.0] * 6])

    loss, _ = grpo_surrogate(logprobs, old, advantages, mask, normalize="sequence")
    loss.backward()
    short, long = logprobs.grad[0, 0].abs(), logprobs.grad[1, 0].abs()
    assert short > long * 2.5, "the short sequence should dominate per token"

    logprobs.grad = None
    loss, _ = grpo_surrogate(logprobs, old, advantages, mask, normalize="token")
    loss.backward()
    assert logprobs.grad[0, 0].abs() == pytest.approx(float(logprobs.grad[1, 0].abs()))


def test_an_unknown_normalisation_is_rejected():
    logprobs, old, advantages, mask = sample()
    with pytest.raises(ValueError):
        grpo_surrogate(logprobs, old, advantages, mask, normalize="per-galaxy")


# --- the KL term ------------------------------------------------------------------


def test_the_kl_estimator_is_non_negative_and_zero_at_equality():
    logprobs, old, advantages, mask = sample()
    ref = logprobs.detach().clone()

    _, same = grpo_surrogate(logprobs, old, advantages, mask, ref_logprobs=ref, beta=0.1)
    assert same["kl"] == pytest.approx(0.0, abs=1e-6)

    _, apart = grpo_surrogate(
        logprobs, old, advantages, mask, ref_logprobs=ref - 0.5, beta=0.1
    )
    assert apart["kl"] > 0.0


def test_beta_zero_ignores_the_reference_entirely():
    logprobs, old, advantages, mask = sample()
    with_ref, _ = grpo_surrogate(
        logprobs, old, advantages, mask, ref_logprobs=torch.randn_like(logprobs), beta=0.0
    )
    without, _ = grpo_surrogate(logprobs, old, advantages, mask, beta=0.0)
    assert torch.allclose(with_ref, without)


# --- the log-prob primitive --------------------------------------------------------


def test_selective_logprobs_matches_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11)
    targets = torch.randint(0, 11, (2, 4))

    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(selective_logprobs(logits, targets), expected, atol=1e-6)


def _tiny_policy():
    """A 2-layer Qwen2 with a LoRA adapter. No download, no GPU, milliseconds."""
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.Qwen2ForCausalLM(config)
    return peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )


def test_one_update_actually_moves_the_policy():
    """The claim the exit gate rests on: the update changes the weights.

    A smoke run can print a loss and change nothing — the first CPU run of
    `grpo_scratch.py` did exactly that, because truncated completions scored
    zero, every group was degenerate, and the gradient was correctly zero. This
    separates "the loop ran" from "the policy moved".
    """
    torch.manual_seed(0)
    policy = _tiny_policy()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert trainable, "LoRA produced no trainable parameters"
    before = [p.detach().clone() for p in trainable]

    sequences = torch.randint(0, 64, (4, 20))
    completion_len = 6
    mask = torch.ones(4, completion_len)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    with torch.no_grad():
        old = completion_logprobs(policy, sequences, completion_len)

    optimizer = torch.optim.AdamW(trainable, lr=0.1)
    logprobs = completion_logprobs(policy, sequences, completion_len)
    loss, _ = grpo_surrogate(logprobs, old, advantages, mask)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in trainable)
    optimizer.step()

    moved = [not torch.equal(b, p.detach()) for b, p in zip(before, trainable)]
    assert any(moved), "optimizer.step() left every parameter unchanged"


def _degenerate_step(weight_decay: float):
    torch.manual_seed(0)
    policy = _tiny_policy()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]

    sequences = torch.randint(0, 64, (4, 20))
    with torch.no_grad():
        old = completion_logprobs(policy, sequences, 6)

    optimizer = torch.optim.AdamW(trainable, lr=0.1, weight_decay=weight_decay)
    logprobs = completion_logprobs(policy, sequences, 6)
    loss, _ = grpo_surrogate(logprobs, old, torch.zeros(4), torch.ones(4, 6))
    loss.backward()
    grad_max = max(float(p.grad.abs().max()) for p in trainable if p.grad is not None)
    optimizer.step()
    return before, trainable, grad_max


def test_a_degenerate_group_produces_no_gradient_at_all():
    before, trainable, grad_max = _degenerate_step(weight_decay=0.0)
    assert grad_max == 0.0
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, trainable))


def test_weight_decay_would_move_the_policy_even_with_no_gradient():
    """Why `Config.weight_decay` is 0.0 rather than AdamW's default 0.01.

    Decay is applied whether or not the reward said anything, so on a batch of
    degenerate groups — 16% of groups at the frozen baseline — it shrinks the
    adapter for reasons unrelated to the reward. In an experiment about
    attributing reward changes, that is a confound rather than a regulariser.
    """
    before, trainable, grad_max = _degenerate_step(weight_decay=0.01)
    assert grad_max == 0.0, "the gradient is still exactly zero"
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, trainable)
    ), "decay moved the policy anyway — this is the behaviour Config disables"


def test_groups_of_different_prompt_lengths_can_be_updated_together():
    """The crash the first real run hit, and the reason for not padding around it.

    Two prompts tokenize to different lengths, so their groups cannot be
    concatenated. Padding them to a common width would work and would also
    corrupt the log-probs, since completion_logprobs passes no attention mask
    and a padded row attends to the pads. The update iterates groups instead.
    """
    torch.manual_seed(0)
    policy = _tiny_policy()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05, weight_decay=0.0)
    before = [p.detach().clone() for p in trainable]

    completion_len = 5
    groups = [
        (torch.randint(0, 64, (3, prompt_len + completion_len)), advantage)
        for prompt_len, advantage in ((11, [1.0, -1.0, 0.0]), (17, [0.5, 0.5, -1.0]))
    ]
    assert groups[0][0].shape[1] != groups[1][0].shape[1], "widths must actually differ"
    with pytest.raises(RuntimeError):
        torch.cat([g[0] for g in groups], dim=0)

    total = sum(g[0].shape[0] for g in groups)
    optimizer.zero_grad(set_to_none=True)
    for sequences, adv in groups:
        mask = torch.ones(sequences.shape[0], completion_len)
        with torch.no_grad():
            old = completion_logprobs(policy, sequences, completion_len)
        logprobs = completion_logprobs(policy, sequences, completion_len)
        loss, _ = grpo_surrogate(logprobs, old, torch.tensor(adv), mask)
        (loss * sequences.shape[0] / total).backward()
    optimizer.step()

    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, trainable))


@pytest.mark.parametrize(
    "predicts,confident",
    [
        ((5, 6), True),  # the actual completion tokens
        ((4, 5), False),  # shifted by one -- what an off-by-one would score
    ],
)
def test_completion_logprobs_scores_the_sampled_tokens(predicts, confident):
    """Decisive on alignment: an off-by-one would make the wrong stub confident.

    For `[1,2,3,4,5,6]` with `completion_len=2`, the tokens that were sampled
    are 5 and 6, so a model betting on 5 then 6 must score near zero and one
    betting on 4 then 5 must not.
    """

    class Stub:
        vocab = 8

        def __call__(self, input_ids, logits_to_keep):
            batch = input_ids.shape[0]
            logits = torch.full((batch, logits_to_keep, self.vocab), -20.0)
            # logits_to_keep is completion_len + 1; completion_logprobs drops the
            # last row, so rows 0..completion_len-1 are the ones that get scored.
            for t, token in enumerate(predicts):
                logits[:, t, token] = 20.0
            return type("Out", (), {"logits": logits})()

    got = completion_logprobs(Stub(), torch.tensor([[1, 2, 3, 4, 5, 6]]), completion_len=2)

    assert got.shape == (1, 2)
    if confident:
        assert got.max() > -0.01, "betting on the sampled tokens should score ~0"
    else:
        assert got.max() < -10.0, "a one-token shift must not look confident"
