#!/usr/bin/env python
"""Notebook 08 -- a critic that is another model, and progress measured under a prover.

07's critic was a linear head on the policy's own hidden states (layer 23). Its PPO run from the recovery seed was
the series' first to lift a greedy evaluation above step 0 (0.586 -> 0.658 at step 40, 60 dev questions). This
notebook keeps 07's seed, task, reward and PPO configuration and replaces the critic, following Setlur et al.,
"Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning" (Google DeepMind / Google Research and
CMU; arXiv 2410.08146, ICLR 2025):

  * a step's process reward should be progress -- the step's advantage, the change in the chance of ending right
    before and after it -- not the value of the state it reaches;
  * progress should be measured under a *prover* policy distinct from the base policy. Good provers are
    complementary to it: their advantages tell the base policy's steps apart (variance under its actions) while
    staying aligned with its own (correlation); provers too strong (succeed from anywhere) or too weak (fail from
    anywhere) give advantages near zero. Best-of-K of the base policy, BoK(pi), is the paper's working class (K = 4
    throughout its RL), with Q^{BoK(pi)} = 1 - (1 - Q^pi)^K (its Eq. 30); across models a prover one size up was best
    (2B -> 9B, 9B -> 27B), and a weaker 9B prover beat the 27B for a 27B base;
  * online RL takes the effective reward Q^pi + alpha * A^mu per step, A^mu predicted by a separate verifier model.

What 08 changes:
  * The critic is a separate pretrained model, frozen, read at every turn boundary of an episode's text -- not a
    head on the policy. The paper's verifiers were a separate model too (all fine-tuned from Gemma 9B, for 2B, 9B and
    27B policies); fine-tuning one inside each PPO step is beyond this machine, so here two linear heads on the frozen
    model's features are refitted from scratch at every step on the last 25 steps' turns: v (ridge regression on the
    turn's return to go: PPO's baseline, the job 07's head did) and q (L2 logistic regression on the episode's
    success -- the decision right with the proving simulations run: the base policy's Q^pi). Candidates: Qwen3-1.7B
    (another family, tokenizer and architecture, 2x the policy), Qwen3.5-2B (the policy's family one size up) and
    Qwen3-4B (the paper's verifier-to-policy ratio, 4.5x). ``critic-fit`` reads each, and 07's own critic's features,
    over 07's sampled PPO episodes and compares them as PPO will use them (fitted on 25 steps, tested on the next)
    and on held-out dev questions; then the paper's prover criterion (distinguishability + alignment, its Theorem
    3.1) chooses K.
  * The prover is BoK(pi): A^mu of a turn = f_K(q after it) - f_K(q before it), f_K(q) = 1 - (1 - q)^K, the last
    turn's "after" being the episode's own outcome; every turn's advantage gains alpha * A^mu (``--prover-alpha``),
    its own A^mu only, as the paper's per-step effective reward Q^pi + alpha A^mu.
  * Everything else -- the seed (07's seed-n200-r100), the turn as the unit of credit, lr 2e-6, KL 0.01, the gates,
    the evaluations -- is 07's, copied (no code is shared between notebooks). 07's questions, references and
    recovery episodes are read in place, and so is its seed. The first 25 steps are 07's own (``--warm-from``): the
    same seed draws the same questions and the policy is not updated in them, so their episodes fill the new
    critic's window -- and the run stays step for step comparable with 07's.
  * Three of the 37 PPO implementation details (Huang et al. 2022), as settings that default to 03-07's update:
    advantage normalization (``--adv-norm`` batch / minibatch), minibatch updates (``--minibatches``), a KL early stop
    (``--target-kl``; the approximate KL to the rollout policy is logged either way), and the details' learning-rate
    annealing and gradient clip (``--lr-anneal``, ``--max-grad-norm``). ``08_prover_critic_ppo_run.sh`` with D37=1
    takes the details' defaults where they carry over.
  * Generation on Apple silicon by mlx-lm (``--generator mlx``, for ``ppo`` and ``probe``): ``MLXPolicy`` folds the
    policy's LoRA into the base weights before each rollout, ``rollout_mlx`` is ``rollout`` generated there; the
    update and every log-probability stay PyTorch's. ``--save-every`` saves the continuation state between
    evaluations too.

Nothing here trains on the MembraneClaw benchmark (D1-D6): it is evaluation-only. Every gold answer comes from
running the real simulator on the generated scenario, through the same argument handling the deployed MCP server
applies.

Subcommands (the rollouts' tools run in .venv-watertap worker processes):

  critic-fit  the candidate critics fitted and compared on 07's sampled episodes (online replay; held-out dev
              questions), and the paper's prover criterion over K; writes 08_prover_critic_critic_fit.json
  ppo         07's PPO with the separate critic and the prover's advantage (--critic, --critic-layer, --prover-k,
              --prover-alpha, --warm-from); --critic policy is 07's own critic, as a control
  probe       07's probe (and --forced 1), for 08's checkpoints (--generator mlx: generated by mlx-lm; 07's adapters
              too, the harness being 07's)
  worker      JSON-lines tool server for rollouts (deployed argument handling, no rate limit)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 07: on Apple silicon the GPU allocates from the machine's own memory, and PyTorch's MPS allocator may by default
# grow to 1.7x the recommended working set -- 88 GB on this 64 GB machine. The first seed run did: its cached
# buffers of every episode length reached 72 GB, the machine swapped, and single SFT steps took 13 minutes. The
# allocator is capped at 0.7 of the working set (36 GB here). Its low watermark -- above which it commits command
# buffers early and collects garbage on every allocation -- is off: at 0.4 it made a decode step 3x slower
# (256 ms against 83 ms, 8 rows at 5.6k tokens); this file empties the cache itself (``release_device_cache``).
# Set before torch is imported (every torch import in this file is inside a function).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

HERE = Path(__file__).resolve().parent
PREFIX = "08_prover_critic"
DATA = HERE / "07_recover_from_errors_data"        # 07's recovery episodes (the forced probe's starting turns), read in place
QDATA = HERE / "06_direction_and_compare_data"     # the questions and the blind reference episodes: 06's, read in place
RUNS07 = HERE / "07_recover_from_errors_runs"      # 07's seed (08's start) and its sampled PPO episodes (critic-fit's data)
#: 08: the other notebook 07 (Rui Tong's, commit effa2e4 on experiment/smoke-test35-07-error-recovery,
#: 07_error_recovery_data/train_reference_recovery.jsonl): a recovery episode for every train question -- one slip (an
#: input left unconverted, else a x10 / x0.1 factor), the tool's real error, the fix -- converted to this file's form
#: (its slipped turn's ``"context": true`` is ``"label": false`` here). ``seed --recovery-every`` uses them.
RT_RECOVERY = HERE / f"{PREFIX}_data" / "train_recovery_effa2e4.jsonl"
TOOLS_JSON = HERE / f"{PREFIX}_tools.json"
REPO = HERE.parents[2]
MCP_DIR = REPO / "mcp_watertap"
AGENT_CORE = REPO / "agent" / "core.py"
BENCH_DIR = REPO / "experiments" / "auto-evaluate" / "benchmarks" / "Datasets Harness"

RO, SWRO, CONVERT, COMPARE = "simulate_ro", "simulate_swro_system", "convert_units", "check_limits"
TOOLS = (RO, SWRO)                # the simulators, declared exactly as the deployed server serves them
LOCAL_TOOLS = (CONVERT, COMPARE)  # this notebook's own tools: served by the harness, not on the deployed server
ALL_TOOLS = TOOLS + LOCAL_TOOLS
#: ``--compare 0`` (reference, check-harness, seed, probe, ppo): 06's questions with 05's episode -- no
#: check_limits declared, no check turn, the comparison lines written by the policy. The "direction
#: sentences alone" row; its references are ``train_reference_nocompare.jsonl``.
COMPARE_ON = True


def reference_path(split: str = "train") -> Path:
    """06's blind reference episodes (07 reads them; ``reference`` without --recover rebuilds them here)."""
    own = DATA / f"{split}_reference{'' if COMPARE_ON else '_nocompare'}.jsonl"
    return own if own.exists() else QDATA / own.name


def recovery_path(split: str = "train") -> Path:
    """07's recovery episodes of a split (``reference --recover 1``)."""
    return DATA / f"{split}_recovery.jsonl"


def pick_device() -> str:
    """CUDA if there is a card, else Apple silicon's MPS, else the CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_rng_state(device: str):
    import torch
    if device == "cuda":
        return torch.cuda.get_rng_state_all()
    if device == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device: str, state) -> None:
    import torch
    if state is None:
        return
    if device == "cuda":
        torch.cuda.set_rng_state_all(state)
    elif device == "mps":
        torch.mps.set_rng_state(state)


def cap_device_memory(max_vram_mib: int) -> None:
    """06's per-process cap on the card (13,000 MiB of 16 GB); CUDA only -- MPS shares the machine's memory."""
    import torch
    if max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_mib * 2**20 / total))


def peak_device_mib() -> float | None:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def release_device_cache() -> None:
    """07: hand the MPS allocator's cached buffers back (episode lengths vary, so a cached buffer is rarely the size
    the next episode asks for). Called after each optimizer step, generate call and evaluation; no-op elsewhere."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


#: 07: on MPS every sequence length an op meets is compiled and cached for good (PyTorch's MPS graph cache has no
#: eviction), so a process that sees a new length per episode grows without bound: the second seed run went from
#: 13 GB at episode 83 to 26 GB at episode 292, slowing as it neared the allocator's cap. Lengths are rounded up to
#: a multiple of this on MPS (right padding unlabelled in SFT; more left padding, masked, in generation and in PPO's
#: batches), which leaves a handful of lengths. CUDA and CPU keep 06's exact lengths.
SHAPE_BUCKET_MPS = 512


def bucketed(n: int, device) -> int:
    """``n`` rounded up to a multiple of SHAPE_BUCKET_MPS on MPS; ``n`` elsewhere."""
    return -(-n // SHAPE_BUCKET_MPS) * SHAPE_BUCKET_MPS if str(device).startswith("mps") else n


def _inverse_unit_lower(a):
    """(I + N)^-1 for N the strictly lower triangle of ``a`` (batched C x C, C a power of two; the diagonal is
    ignored, as solve_triangular's ``unitriangular=True`` ignores it), by recursive doubling: the inverse of the
    2s x 2s diagonal blocks from the inverses of their two s x s halves, [[A, 0], [L, B]]^-1 = [[A^-1, 0],
    [-B^-1 L A^-1, B^-1]], every block of a size at once -- log2(C) rounds of batched matmuls, nothing indexed
    in place (on MPS a sliced or in-place step makes the backward copy whole tensors once per step). This is
    the recursive triangular inversion LAPACK's trtri uses. (The product (I - N)(I + N^2)(I + N^4)... is exact
    in exact arithmetic, but N's powers overflow float32 on real episodes: the log-probs came out NaN.)"""
    import torch
    n = a.tril(-1)
    size, total = 1, a.shape[-1]
    inv = torch.ones(*a.shape[:-2], total, 1, 1, dtype=a.dtype, device=a.device)  # the 1 x 1 diagonal blocks
    while size < total:
        blocks = n.reshape(*n.shape[:-2], total // (2 * size), 2 * size, total // (2 * size), 2 * size)
        lower = torch.diagonal(blocks, dim1=-4, dim2=-2).movedim(-1, -3)[..., size:, :size]  # each pair's L
        top, bottom = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        low = -(bottom @ lower @ top)
        inv = torch.cat([torch.cat([top, torch.zeros_like(top)], dim=-1), torch.cat([low, bottom], dim=-1)], dim=-2)
        size *= 2
    return inv.squeeze(-3)


def mps_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int = 64, initial_state=None,
                               output_final_state: bool = False, use_qk_l2norm_in_kernel: bool = False, **kwargs):
    """transformers' torch_chunk_gated_delta_rule (Qwen3.5's linear-attention layers without the fla kernels),
    the same computation, with the UT system inverted by ``_inverse_unit_lower`` instead of solved by
    ``torch.linalg.solve_triangular``. On MPS the triangular solve is 0.229 s of the rule's 0.234 s per layer on
    a 6k-token episode (18 layers: 4.2 s of a 5.0 s forward); forward substitution is 0.013 s. Installed only on
    MPS (``patch_delta_rule``); CUDA and CPU keep transformers' own."""
    import torch
    import torch.nn.functional as F
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    num_chunks = decay.shape[2]
    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strictly_upper, float("-inf")).exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    inv = _inverse_unit_lower(ut_system)
    new_values, k_cumdecay = inv @ v_beta, inv @ (k_beta * cum_decay.exp().unsqueeze(-1))
    state = (torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=new_values.dtype, device=new_values.device)
             if initial_state is None else initial_state.to(new_values))
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    # The scan over chunks, on per-chunk views taken once (``unbind``): indexing chunk i inside the loop makes
    # the backward write a whole-size gradient per chunk.
    nv, kc, qs, ia, ks, cd = (x.unbind(2) for x in (new_values, k_cumdecay, query, intra, key, chunk_decay))
    outs = []
    for i in range(num_chunks):
        v_new = nv[i] - kc[i] @ state
        outs.append(qs[i] @ state + ia[i] @ v_new)
        state = state * cd[i] + ks[i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def patch_delta_rule(device: str) -> None:
    """On MPS, Qwen3.5's linear-attention layers call ``mps_chunk_gated_delta_rule`` (see there)."""
    if device != "mps":
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    qwen35.torch_chunk_gated_delta_rule = mps_chunk_gated_delta_rule

# ---------------------------------------------------------------------------------------------
# The fixed parts of every episode: system prompt, tool defaults, the metrics an answer may cite.
# ---------------------------------------------------------------------------------------------

#: agent/core.py's SYSTEM_MESSAGE, verbatim (``check`` asserts it still matches).
DEPLOYED_SYSTEM_MESSAGE = (
    "You are a capable assistant running on a local Qwen3.5 deployment. "
    "Use the provided tools when they give you a more accurate answer than reasoning alone; "
    "otherwise answer directly. When documents are attached, ground your answer in them and "
    "say so if the answer is not present."
)
#: The one addition to the deployed prompt: an answer contract that code can grade.
ANSWER_CONTRACT = (
    "Finish your reply with one JSON object and nothing after it:\n"
    '{"decision": ..., "violations": [...], "values": {...}}\n'
    '- "decision": the direct answer to the question. "pass" or "fail" when asked whether '
    "something meets its limits; otherwise the option or setting you recommend, written the way "
    "the user named it; null if none of the options works.\n"
    '- "violations": when asked whether something meets its limits, the tool-result fields whose '
    "limits are broken; otherwise [].\n"
    '- "values": the numbers the user asked for, keyed by the tool-result field they come from '
    "as section.field, in that field's units."
)
SYSTEM_PROMPT = DEPLOYED_SYSTEM_MESSAGE + "\n\n" + ANSWER_CONTRACT

#: mcp_watertap ro_model.DEFAULTS / swro_model.DEFAULTS (``generate`` asserts they still match).
DEFAULTS: dict[str, dict[str, Any]] = {
    RO: {
        "feed_flow_mass_kg_s": 1.0, "feed_nacl_mass_frac": 0.035, "feed_pressure_bar": 50.0,
        "feed_temperature_c": 25.0, "membrane_area_m2": 50.0, "A_comp": 4.2e-12, "B_comp": 3.5e-8,
        "permeate_pressure_bar": 1.01325, "pressure_drop_bar": 3.0, "channel_height_m": 0.002,
        "spacer_porosity": 0.75, "module_length_m": 20.0, "cp_modulus": 1.1,
        "mass_transfer_coeff": 2e-5, "concentration_polarization": "calculated",
        "mass_transfer_coefficient": "calculated",
    },
    SWRO: {
        "erd_type": "pressure_exchanger", "feed_flow_m3_s": 0.3092, "feed_tds_g_L": 35.0,
        "feed_tss_g_L": 0.03, "feed_temperature_c": 24.85, "ro_area_m2": None, "A_comp": 4.2e-12,
        "B_comp": 3.5e-8, "p1_pressure_bar": 70.0, "p1_efficiency": 0.80, "pxr_efficiency": 0.95,
        "p2_efficiency": 0.80, "erd_efficiency": 0.95,
    },
}

#: Result fields an answer may cite, as section.field paths into the tool's JSON.
METRICS = {
    RO: ("permeate.flow_kg_s", "permeate.nacl_ppm", "performance.water_recovery_pct",
         "performance.salt_rejection_pct", "flux.water_LMH"),
    SWRO: ("costing.specific_energy_kWh_m3", "costing.LCOW_usd_m3",
           "performance.product_flow_m3_s", "performance.system_recovery_pct"),
}
#: Smallest |value - threshold| / value any gold point may have. A 0.5% error in one input moves
#: RO outputs by at most 1.43% (measured over 12 scenarios x 7 inputs), so 2% keeps every gold
#: pass/fail stable under the argument tolerance; rejection moves 0.02%, hence its smaller margin.
MARGIN = defaultdict(lambda: 0.02, {"performance.salt_rejection_pct": 0.001})
#: How far past the nearest value an open-ended threshold may sit (keeps limits plausible).
SPREAD = {
    "permeate.flow_kg_s": 0.25, "permeate.nacl_ppm": 0.30, "performance.water_recovery_pct": 0.15,
    "performance.salt_rejection_pct": 0.006, "flux.water_LMH": 0.25,
    "costing.specific_energy_kWh_m3": 0.25, "costing.LCOW_usd_m3": 0.25,
    "performance.product_flow_m3_s": 0.25, "performance.system_recovery_pct": 0.15,
}
PCT_METRICS = {"performance.water_recovery_pct", "performance.salt_rejection_pct",
               "performance.system_recovery_pct"}
METRIC_NAMES = {
    "permeate.flow_kg_s": ["permeate flow", "product water flow"],
    "permeate.nacl_ppm": ["permeate NaCl", "permeate salinity"],
    "performance.water_recovery_pct": ["recovery", "water recovery"],
    "performance.salt_rejection_pct": ["salt rejection"],
    "flux.water_LMH": ["water flux", "flux"],
    "costing.specific_energy_kWh_m3": ["specific energy", "energy use per cubic metre"],
    "costing.LCOW_usd_m3": ["levelized cost of water", "water cost per cubic metre"],
    "performance.product_flow_m3_s": ["product flow", "plant output"],
    "performance.system_recovery_pct": ["overall plant recovery", "system recovery"],
}

# ---------------------------------------------------------------------------------------------
# Units. Every quantity is sampled in the user's unit on a round grid and converted to the tool's
# unit exactly, so the gold is unique. Level 0 = the tool's own unit, 1 = a decimal shift or a
# relabel, 2 = a multiplicative conversion. Only conversions with exact factors are used.
# ---------------------------------------------------------------------------------------------

def _u(name: str, level: int, factor: float, grid: float, fmt: str, offset: float = 0.0) -> dict:
    return {"name": name, "level": level, "factor": factor, "offset": offset, "grid": grid, "fmt": fmt}


PSI_PER_BAR = 1e5 / 6894.757293168
FT2_PER_M2 = 1 / 0.09290304
MGD_PER_M3S = 86400 / 3785.411784
GFD_PER_LMH = 0.09290304 * 24 / 3.785411784
INPUT_UNITS = {
    "ro_flow": [_u("kg/s", 0, 1, 0.01, "{:.2f} kg/s"), _u("t/h", 2, 3.6, 0.1, "{:.1f} t/h"),
                _u("kg/h", 2, 3600, 10, "{:,.0f} kg/h")],
    "salinity": [_u("mass fraction", 0, 1, 0.0005, "a NaCl mass fraction of {:.4f}"),
                 _u("g/kg", 1, 1000, 0.1, "{:.1f} g/kg"), _u("permille", 1, 1000, 0.1, "{:.1f} ‰"),
                 _u("pct", 1, 100, 0.01, "{:.2f} % salt by mass"),
                 _u("mg/kg", 1, 1e6, 100, "{:,.0f} mg/kg")],
    "temp": [_u("C", 0, 1, 1, "{:.0f} °C"), _u("F", 2, 1.8, 1, "{:.0f} °F", offset=32)],
    "area": [_u("m2", 0, 1, 5, "{:,.0f} m²"), _u("ft2", 2, FT2_PER_M2, 50, "{:,.0f} ft²")],
    # 1 m/s = 3.6e6 LMH and 1 bar = 1e5 Pa: A in LMH/bar is A in m/s/Pa times 3.6e11 (04 had 3.6e14).
    "A": [_u("SI", 0, 1e12, 0.1, "{:.1f}e-12 m/s/Pa"), _u("LMH/bar", 2, 3.6e11, 0.01, "{:.2f} LMH/bar")],
    "B": [_u("SI", 0, 1e8, 0.1, "{:.1f}e-8 m/s"), _u("LMH", 2, 3.6e6, 0.001, "{:.3f} LMH")],
    "pressure": [_u("bar", 0, 1, 0.5, "{:g} bar"), _u("kPa", 1, 100, 50, "{:,.0f} kPa"),
                 _u("MPa", 1, 0.1, 0.05, "{:.2f} MPa"), _u("psi", 2, PSI_PER_BAR, 5, "{:,.0f} psi")],
    "swro_flow": [_u("m3/s", 0, 1, 0.001, "{:.3f} m³/s"), _u("L/s", 1, 1000, 1, "{:,.0f} L/s"),
                  _u("m3/h", 2, 3600, 10, "{:,.0f} m³/h"), _u("m3/d", 2, 86400, 100, "{:,.0f} m³/d"),
                  _u("MGD", 2, MGD_PER_M3S, 0.01, "{:.2f} MGD")],
    "tds": [_u("g/L", 0, 1, 0.1, "{:.1f} g/L"), _u("mg/L", 1, 1000, 100, "{:,.0f} mg/L"),
            _u("kg/m3", 1, 1, 0.1, "{:.1f} kg/m³")],
    "eff": [_u("fraction", 0, 1, 0.01, "{:.2f}"), _u("pct", 1, 100, 1, "{:.0f} %")],
}
#: Units a threshold on each metric may be stated in ("{}" is filled with a formatted number).
METRIC_UNITS = {
    "permeate.flow_kg_s": [_u("kg/s", 0, 1, 0, "{} kg/s"), _u("m3/h", 2, 3.6, 0, "{} m³/h"),
                           _u("m3/d", 2, 86.4, 0, "{} m³/d"), _u("t/h", 2, 3.6, 0, "{} t/h")],
    "permeate.nacl_ppm": [_u("mg/L", 0, 1, 0, "{} mg/L"), _u("ppm", 0, 1, 0, "{} ppm")],
    "performance.water_recovery_pct": [_u("%", 0, 1, 0, "{} %")],
    "performance.salt_rejection_pct": [_u("%", 0, 1, 0, "{} %")],
    "flux.water_LMH": [_u("LMH", 0, 1, 0, "{} LMH"), _u("L/m2/h", 1, 1, 0, "{} L/m²/h"),
                       _u("gfd", 2, GFD_PER_LMH, 0, "{} gfd")],
    "costing.specific_energy_kWh_m3": [_u("kWh/m3", 0, 1, 0, "{} kWh/m³")],
    "costing.LCOW_usd_m3": [_u("$/m3", 0, 1, 0, "${} per m³"),
                            _u("$/kgal", 2, 3.785411784, 0, "${} per 1,000 gallons")],
    "performance.product_flow_m3_s": [_u("m3/s", 0, 1, 0, "{} m³/s"), _u("L/s", 1, 1000, 0, "{} L/s"),
                                      _u("m3/d", 2, 86400, 0, "{} m³/d"), _u("m3/h", 2, 3600, 0, "{} m³/h"),
                                      _u("MGD", 2, MGD_PER_M3S, 0, "{} MGD")],
    "performance.system_recovery_pct": [_u("%", 0, 1, 0, "{} %")],
}
#: Scan steps in user units, per input unit.
SCAN_STEPS = {
    ("pressure", "bar"): [1.5, 2, 2.5, 3], ("pressure", "kPa"): [150, 200, 250, 300],
    ("pressure", "MPa"): [0.15, 0.2, 0.25, 0.3], ("pressure", "psi"): [25, 30, 40, 50],
    ("area", "m2"): [25, 30, 40, 50], ("area", "ft2"): [300, 400, 500],
}


#: How a unit is written in a convert_units call: the way the prompts write it.
UNIT_SYMBOL = {
    "kg/s": "kg/s", "t/h": "t/h", "kg/h": "kg/h", "mass fraction": "mass fraction", "g/kg": "g/kg",
    "permille": "‰", "pct": "%", "mg/kg": "mg/kg", "C": "C", "F": "°F", "m2": "m²", "ft2": "ft²",
    "SI": "SI", "LMH/bar": "LMH/bar", "LMH": "LMH", "bar": "bar", "kPa": "kPa", "MPa": "MPa", "psi": "psi",
    "m3/s": "m³/s", "L/s": "L/s", "m3/h": "m³/h", "m3/d": "m³/d", "MGD": "MGD", "g/L": "g/L", "mg/L": "mg/L",
    "kg/m3": "kg/m³", "fraction": "fraction", "%": "%", "ppm": "ppm", "L/m2/h": "L/m²/h", "gfd": "gfd",
    "kWh/m3": "kWh/m³", "$/m3": "$/m³", "$/kgal": "$/kgal",
}
#: The simulators' own unit of each input quantity and of each result metric (the convert target).
TOOL_UNIT = {"ro_flow": "kg/s", "salinity": "mass fraction", "temp": "C", "area": "m²", "A": "m/s/Pa",
             "B": "m/s", "pressure": "bar", "swro_flow": "m³/s", "tds": "g/L", "eff": "fraction"}
METRIC_TOOL_UNIT = {"permeate.flow_kg_s": "kg/s", "permeate.nacl_ppm": "mg/L", "flux.water_LMH": "LMH",
                    "costing.LCOW_usd_m3": "$/m³", "performance.product_flow_m3_s": "m³/s",
                    "costing.specific_energy_kWh_m3": "kWh/m³", "performance.water_recovery_pct": "%",
                    "performance.salt_rejection_pct": "%", "performance.system_recovery_pct": "%"}


def unit_spec(q: str, name: str) -> dict:
    return next(u for u in INPUT_UNITS[q] if u["name"] == name)


def metric_unit_spec(path: str, name: str) -> dict:
    return next(u for u in METRIC_UNITS[path] if u["name"] == name)


def needs_conversion(u: dict) -> bool:
    """A stated unit whose number differs from the tool's (level 0 is the tool's own unit; a relabel
    such as L/m²/h for LMH changes nothing)."""
    return u["level"] > 0 and (u["factor"] != 1 or u["offset"] != 0)


def to_user(u: dict, tool_value: float) -> float:
    return tool_value * u["factor"] + u["offset"]


def to_tool(u: dict, user_value: float) -> float:
    return (user_value - u["offset"]) / u["factor"]


def round_grid(x: float, grid: float) -> float:
    return round(round(x / grid) * grid, 10)


def fmt_step(x: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{x:,.{decimals}f}"


# ---------------------------------------------------------------------------------------------
# convert_units: the one tool this notebook adds. Exact factors, the generator's own; every unit
# the prompts use, and the usual spellings of each. Served by ToolHost like the simulators.
# ---------------------------------------------------------------------------------------------

#: kind -> unit -> (multiplier, offset): value_in_base = (value - offset) * multiplier.
UNIT_KINDS: dict[str, dict[str, tuple[float, float]]] = {
    "mass flow": {"kg/s": (1, 0), "t/h": (1 / 3.6, 0), "kg/h": (1 / 3600, 0),
                  # water taken at 1000 kg/m3, as the generator does for permeate flow
                  "m3/h": (1 / 3.6, 0), "m3/d": (1 / 86.4, 0)},
    "volume flow": {"m3/s": (1, 0), "l/s": (1e-3, 0), "m3/h": (1 / 3600, 0), "m3/d": (1 / 86400, 0),
                    "mgd": (1 / MGD_PER_M3S, 0)},
    "salinity": {"mass fraction": (1, 0), "g/kg": (1e-3, 0), "‰": (1e-3, 0), "%": (1e-2, 0),
                 "mg/kg": (1e-6, 0), "ppm": (1e-6, 0)},
    "concentration": {"g/l": (1, 0), "mg/l": (1e-3, 0), "kg/m3": (1, 0), "ppm": (1e-3, 0)},
    "temperature": {"c": (1, 0), "f": (1 / 1.8, 32)},
    "area": {"m2": (1, 0), "ft2": (1 / FT2_PER_M2, 0)},
    "water permeability": {"m/s/pa": (1, 0), "lmh/bar": (1 / 3.6e11, 0)},
    "salt permeability": {"m/s": (1, 0), "lmh": (1 / 3.6e6, 0)},
    "flux": {"lmh": (1, 0), "l/m2/h": (1, 0), "gfd": (1 / GFD_PER_LMH, 0)},
    "pressure": {"bar": (1, 0), "kpa": (1e-2, 0), "mpa": (10, 0), "psi": (1 / PSI_PER_BAR, 0), "pa": (1e-5, 0)},
    "cost": {"$/m3": (1, 0), "$/kgal": (1 / 3.785411784, 0)},
    "efficiency": {"fraction": (1, 0), "%": (1e-2, 0)},
}
_UNIT_SYNONYMS = {
    "degc": "c", "celsius": "c", "°c": "c", "℃": "c", "deg c": "c", "degf": "f", "fahrenheit": "f", "°f": "f",
    "deg f": "f", "sqm": "m2", "sq m": "m2", "m^2": "m2", "sqft": "ft2", "sq ft": "ft2", "ft^2": "ft2",
    "m^3/s": "m3/s", "m^3/h": "m3/h", "m^3/d": "m3/d", "m3/hr": "m3/h", "m3/day": "m3/d", "m3/hour": "m3/h",
    "l/h": None, "lps": "l/s", "liters/s": "l/s", "litres/s": "l/s", "t/hr": "t/h", "tonne/h": "t/h",
    "tonnes/h": "t/h", "ton/h": "t/h", "tons/h": "t/h", "tph": "t/h", "kg/hr": "kg/h", "kg/hour": "kg/h",
    "permille": "‰", "per mille": "‰", "promille": "‰", "percent": "%", "pct": "%", "% salt by mass": "%",
    "w/w": "mass fraction", "massfraction": "mass fraction", "mass frac": "mass fraction",
    "mg/kg": "mg/kg", "l/m2/hr": "l/m2/h", "l/(m2 h)": "l/m2/h", "l/(m2·h)": "l/m2/h", "lm2h": "lmh",
    "gal/ft2/d": "gfd", "gal/ft2/day": "gfd", "gfd": "gfd", "kilopascal": "kpa", "megapascal": "mpa",
    "usd/m3": "$/m3", "$ per m3": "$/m3", "$/1000 gal": "$/kgal", "$/1,000 gal": "$/kgal",
    "$/1,000 gallons": "$/kgal", "$/1000 gallons": "$/kgal", "$/1,000 gal": "$/kgal", "$/kgal": "$/kgal",
    "usd/kgal": "$/kgal", "$/thousand gallons": "$/kgal",
    "$/1000gal": "$/kgal", "lmh/bar": "lmh/bar", "l/m2/h/bar": "lmh/bar", "m/s/pa": "m/s/pa", "m/(s pa)": "m/s/pa",
    "m/s": "m/s", "mgd": "mgd", "million gallons per day": "mgd",
}


def _norm_unit(text: str) -> str:
    t = str(text).strip().lower().replace("³", "3").replace("²", "2").replace("μ", "u").replace("·", " ")
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+per\s+", "/", t)
    t = re.sub(r"\s+", " ", t)
    if t.startswith("degrees "):
        t = t[8:]
    t = t.replace("°", "")
    return _UNIT_SYNONYMS.get(t, t) or t


def convert_units(value: Any, from_unit: str, to_unit: str) -> str:
    """The tool: JSON {"value": ..., "unit": ...}, or {"error": ...}."""
    x = parse_number(value)
    if x is None:
        return json.dumps({"error": f"value must be a number, got {value!r}"})
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f == t:
        return json.dumps({"value": float(f"{x:.6g}"), "unit": to_unit}, ensure_ascii=False)
    kinds = [k for k, units in UNIT_KINDS.items() if f in units and t in units]
    if not kinds:
        known = sorted({u for units in UNIT_KINDS.values() for u in units})
        return json.dumps({"error": f"cannot convert {from_unit!r} to {to_unit!r}; units: {', '.join(known)}"})
    (mf, of), (mt, ot) = UNIT_KINDS[kinds[0]][f], UNIT_KINDS[kinds[0]][t]
    base = (x - of) * mf
    return json.dumps({"value": float(f"{base / mt + ot:.6g}"), "unit": to_unit}, ensure_ascii=False)


CONVERT_DECL = {"type": "function", "function": {
    "name": CONVERT,
    "description": (
        "Convert one number between units.\n\n"
        "Supported: mass flow (kg/s, t/h, kg/h; m3/h and m3/d of water at 1000 kg/m3), volume flow "
        "(m3/s, L/s, m3/h, m3/d, MGD), salinity as a mass fraction (mass fraction, g/kg, \u2030, %, mg/kg, "
        "ppm), concentration (g/L, mg/L, kg/m3, ppm), temperature (C, F), area (m2, ft2), pressure "
        "(bar, kPa, MPa, psi), water permeability A (m/s/Pa, LMH/bar), salt permeability B (m/s, LMH), "
        "flux (LMH, gfd), cost ($/m3, $/kgal), efficiency (fraction, %).\n\n"
        "Returns JSON {\"value\": ..., \"unit\": ...}. On failure returns {\"error\": ...} rather than raising.\n"),
    "parameters": {"properties": {"value": {"title": "Value", "type": "number"},
                                  "from_unit": {"title": "From Unit", "type": "string"},
                                  "to_unit": {"title": "To Unit", "type": "string"}},
                   "required": ["value", "from_unit", "to_unit"], "title": "convert_unitsArguments",
                   "type": "object"}}}


COMPARE_OPS = {"<=": lambda v, l: v <= l, ">=": lambda v, l: v >= l, "<": lambda v, l: v < l, ">": lambda v, l: v > l}


_LIMIT_SPEC = re.compile(r"\s*(<=|>=|<|>|≤|≥|=<|=>)\s*(-?[\d.,]+(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def limit_in_field_unit(field: str, number: float, unit: str) -> tuple[float | None, str | None]:
    """A limit stated as ``number unit`` in the field's own unit (converted with convert_units' factors
    when the unit differs): (value, None) or (None, error)."""
    if not unit:
        return number, None
    own = METRIC_TOOL_UNIT.get(field)
    if own is None:
        return None, f"{field!r} is not a known field: give its limit without a unit, in the value's own unit"
    got = json.loads(convert_units(number, unit, own))
    if "error" in got:
        return None, got["error"]
    return got["value"], None


def check_limits(limits: Any, values: Any) -> str:
    """The tool, one call per question. ``limits`` = {field: "op number unit"}, the number as the user
    stated it (the tool converts it to the field's own unit); ``values`` = {candidate: {field: number}}.
    Returns a JSON list with one line per candidate -- "candidate: field value meets|breaks op limit;
    ... -> pass|fail", the limit in the field's own unit, a field given without a limit written out as
    "field value". A limit the tool cannot read -- an unknown field, a unit that does not convert to the
    field's own, no "op number" -- is written into every line as "field value (...)" and left out of the
    verdict; a candidate without a number for a limited field gets "field ? (no value)". The verdict is
    fail when a read limit breaks, pass when every limit is read and met, unchecked otherwise. A slip
    costs that limit, not the call: {"error": ...} only when limits or values are not objects of that
    shape."""
    if isinstance(limits, str):
        try:
            limits = json.loads(limits)
        except ValueError:
            pass
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except ValueError:
            pass
    if not isinstance(limits, dict) or not limits:
        return json.dumps({"error": f"limits must be an object of field: \"op number unit\", got {limits!r}"})
    if not isinstance(values, dict) or not values:
        return json.dumps({"error": f"values must be an object of candidate: {{field: number}}, got {values!r}"})
    parsed: dict[str, tuple[str | None, Any]] = {}  # field -> (op, limit) when read, (None, note) when not
    for field, spec in limits.items():
        field = str(field)
        m = _LIMIT_SPEC.fullmatch(str(spec))
        num = parse_number(m.group(2)) if m else None
        if m is None or num is None:
            parsed[field] = (None, f"limit {str(spec)!r} not read")
            continue
        o = m.group(1).replace("≤", "<=").replace("≥", ">=").replace("=<", "<=").replace("=>", ">=")
        own = METRIC_TOOL_UNIT.get(field)
        if own is None:
            parsed[field] = (None, "unknown field")
            continue
        lim, err = limit_in_field_unit(field, num, m.group(3))
        if err:
            parsed[field] = (None, f"limit {str(spec)!r} not in {own}")
            continue
        parsed[field] = (o, lim)
    lines = []
    for label, vals in values.items():
        if not isinstance(vals, dict):
            return json.dumps({"error": f"the values of {label!r} must be an object of field: number, got {vals!r}"})
        parts, broken, unchecked = [], False, False
        for field, (o, lim) in parsed.items():
            x = parse_number(vals[field]) if field in vals else None
            if o is None:
                parts.append(f"{field} {_fmt(x) if x is not None else '?'} ({lim})")
                unchecked = True
            elif x is None:
                parts.append(f"{field} ? (no value)")
                unchecked = True
            else:
                meets = COMPARE_OPS[o](x, lim)
                broken |= not meets
                parts.append(f"{field} {_fmt(x)} {'meets' if meets else 'breaks'} {o} {lim:g}")
        for field, v in vals.items():  # a field without a limit: written out
            if str(field) not in parsed:
                x = parse_number(v)
                parts.append(f"{field} {_fmt(x) if x is not None else '? (no value)'}")
        lines.append(f"{label}: " + "; ".join(parts) + f" -> {'fail' if broken else 'unchecked' if unchecked else 'pass'}")
    return json.dumps(lines, ensure_ascii=False)


COMPARE_DECL = {"type": "function", "function": {
    "name": COMPARE,
    "description": (
        "Check every candidate's values against the limits and write out each comparison.\n\n"
        "limits is an object with one limit per field, written \"op number unit\": op is \">=\" when the "
        "value must be at least the number, \"<=\" when it must be at most the number; \"<\" and \">\" "
        "are strict. A value equal to the number meets \">=\" and \"<=\". The unit is the one the number "
        "is stated in; when it differs from the field's own unit (the unit in the field's name) the number "
        "is converted, with the units convert_units knows. values is an object: for each candidate, by "
        "name, an object with that candidate's value of every limited field, in the field's own unit; a "
        "field given without a limit is written out as it is.\n\n"
        "Returns a JSON list with one line per candidate: \"candidate: field value meets|breaks op limit; "
        "... -> pass|fail\", the limit in the field's own unit. A limit it cannot read -- an unknown field, a "
        "unit that does not convert to the field's own, no op or number -- is written into every line as "
        "\"field value (...)\" and left out of the verdict, and a candidate without a number for a limited "
        "field gets \"field ? (no value)\"; the verdict is then unchecked unless a read limit breaks. Returns "
        "{\"error\": ...} only when limits or values are not objects of that shape.\n"),
    "parameters": {"properties": {"limits": {"title": "Limits", "type": "object",
                                             "additionalProperties": {"type": "string"}},
                                  "values": {"title": "Values", "type": "object",
                                             "additionalProperties": {"type": "object",
                                                                      "additionalProperties": {"type": "number"}}}},
                   "required": ["limits", "values"], "title": "check_limitsArguments", "type": "object"}}}


def nice_in(lo: float, hi: float, rng: random.Random) -> tuple[float, float] | None:
    """A round number in [lo, hi], preferring coarse steps: (value, step)."""
    if not (hi >= lo > 0):
        return None
    top = math.floor(math.log10(hi))
    for exp in range(top, top - 9, -1):
        for mult in (5, 2, 1):
            step = mult * 10.0 ** exp
            a, b = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
            if a <= b:
                return round(rng.randint(a, min(b, a + 3)) * step, 10), step
    return None


# ---------------------------------------------------------------------------------------------
# Phrase banks: (train/dev/test phrasings, holdout_shift-only phrasings).
# ---------------------------------------------------------------------------------------------

OPENERS = {
    "ro_check": (["Quick check on one of our seawater RO trains.",
                  "Can you sanity-check this operating point for me?",
                  "Night shift here, I want to confirm we're inside our limits.",
                  "We're reviewing a setpoint before handing it to operations.",
                  "Need a compliance check on one RO unit.",
                  "I'm looking at one of the RO trains and want a second opinion."],
                 ["Operations flagged this train for a limits check.",
                  "Before tomorrow's audit, please verify this unit."]),
    "ro_scan_pressure": (["We need to pick a pump setpoint for one of our RO trains.",
                          "Feed conditions changed and we're re-tuning the high-pressure pump.",
                          "We're commissioning a new train and choosing its operating pressure.",
                          "Our pump runs on a VFD and we want to settle on a setpoint."],
                         ["The pump contractor is asking us for a setpoint.",
                          "Seasonal re-tuning time for the high-pressure pump."]),
    "ro_scan_area": (["We're sizing the membrane area for an RO train.",
                      "Design question: how much membrane should this train get?",
                      "We're deciding how many elements to load into a new train.",
                      "Expansion project, and the membrane area isn't fixed yet."],
                     ["The design review wants the membrane area settled.",
                      "We have to order elements soon and need the area."]),
    "ro_membrane": (["Procurement sent us membrane offers for a retrofit.",
                     "We're replacing the elements on one train and have a shortlist.",
                     "Several suppliers quoted replacement membranes.",
                     "We have to choose replacement elements for an existing train."],
                    ["Vendor bids came in for the replacement membranes.",
                     "The element tender closed and we have the finalists."]),
    "swro_check": (["Looking at the whole desalination plant now.",
                    "Management wants plant-level energy and cost numbers.",
                    "Plant-wide check, please.",
                    "I need the economics of our seawater plant as it runs today."],
                   ["Finance is asking whether the plant hits its targets.",
                    "Board meeting prep: plant energy and cost."]),
    "swro_option": (["We're comparing configurations for a seawater RO plant.",
                     "We have a few options for the plant's pump and energy-recovery setup.",
                     "Choosing between plant configurations.",
                     "Our consultant laid out some plant configurations."],
                    ["Several plant configurations are on the table.",
                     "The EPC contractor gave us alternatives to compare."]),
}
FACTS = {
    "feed_flow_mass_kg_s": (["Feed flow to the train is {v}.", "The train gets {v} of feed.",
                             "We feed it {v}.", "It runs on {v} of seawater feed."],
                            ["Intake delivers {v} to this train.", "Feed rate sits at {v}."]),
    "feed_nacl_mass_frac": (["Feed salinity is {v}.", "The seawater comes in at {v}.",
                             "The lab puts the feed at {v}.", "Our intake water measures {v}."],
                            ["The latest grab sample reads {v}.", "Salinity at the intake is {v}."]),
    "feed_temperature_c": (["Water temperature is {v}.", "The feed is at {v}.",
                            "It's {v} at the intake.", "Feed temperature is {v}."],
                           ["We're seeing {v} water today.", "Intake temperature is holding at {v}."]),
    "membrane_area_m2": (["The train has {v} of membrane.", "Installed membrane area is {v}.",
                          "There's {v} of active membrane area.", "The rack holds {v} of membrane."],
                         ["Total membrane area comes to {v}.", "It's built with {v} of membrane."]),
    "AB": (["The elements have a water permeability of {A} and a salt permeability of {B}.",
            "Membrane specs: A = {A}, B = {B}.", "The datasheet gives A as {A} and B as {B}.",
            "Water permeability is {A}; salt permeability is {B}."],
           ["Per the vendor, water permeability is {A} and salt permeability {B}.",
            "The elements are rated at A = {A} and B = {B}."]),
    "feed_pressure_bar": (["The high-pressure pump delivers {v}.", "Feed pressure is {v}.",
                           "We run it at {v}.", "Pump discharge is {v}."],
                          ["Pressure at the membrane inlet is {v}.", "The pump is set to {v}."]),
    "feed_flow_m3_s": (["The plant takes in {v} of seawater.", "Plant feed is {v}.",
                        "Intake flow to the plant is {v}.", "We treat {v} of seawater."],
                       ["The intake pumps supply {v}.", "Raw-water flow is {v}."]),
    "feed_tds_g_L": (["Feed TDS is {v}.", "The seawater carries {v} of dissolved solids.",
                      "Raw water TDS is {v}."], ["Dissolved solids in the feed come to {v}."]),
    "swro_temp": (["Seawater temperature is {v}.", "The intake is at {v}.", "Feed temperature is {v}."],
                  ["The sea is at {v} this month."]),
    "p1_pressure_bar": (["The high-pressure pump runs at {v}.", "Pump discharge pressure is {v}.",
                         "The HP pump is set to {v}."], ["The main pump delivers {v}."]),
    "p1_efficiency": (["The high-pressure pump is {v} efficient.", "Pump efficiency is {v}."],
                      ["The HP pump runs at {v} efficiency."]),
    "erd_px": (["Brine energy goes back through a pressure exchanger at {v} efficiency.",
                "Energy recovery is a pressure exchanger, {v} efficient."],
               ["A pressure exchanger recovers the brine energy at {v} efficiency."]),
    "erd_pat": (["Brine energy is recovered by a pump-as-turbine at {v} efficiency.",
                 "The energy-recovery device is a pump-as-turbine running at {v} efficiency."],
                ["A pump-as-turbine on the brine line recovers energy at {v} efficiency."]),
}
DISTURBANCE = (["{name} just moved from {old} to {new}.", "{name} went from {old} to {new} after {event}.",
                "Since {event}, {lname} is {new} (it was {old})."],
               ["{name} shifted from {old} to {new} overnight."])
DISTURB_NAMES = {"feed_nacl_mass_frac": "Feed salinity", "feed_temperature_c": "Water temperature",
                 "feed_flow_mass_kg_s": "Feed flow"}
EVENTS = {"feed_nacl_mass_frac": ["the storm", "the tide turned", "the dry spell set in", "the river plume moved off",
                                  "the spring tide"],
          "feed_temperature_c": ["the cold front came through", "the heatwave started", "the upwelling started",
                                 "the storm", "the seasonal turnover"],
          "feed_flow_mass_kg_s": ["the intake screen clogged", "one intake pump tripped", "the strainer fouled",
                                  "the storm", "the bypass opened"]}
PROPOSAL = (["We're thinking of running the pump at {v}.", "The proposal is to set feed pressure to {v}."],
            ["Operations wants to try {v} on the pump."])
CONSTRAINTS = {
    ("permeate.flow_kg_s", ">="): (["We need at least {v} of permeate.",
                                    "Production can't drop below {v} of permeate.",
                                    "The permeate target is {v} minimum."],
                                   ["Downstream needs no less than {v} of product water."]),
    ("permeate.nacl_ppm", "<="): (["Permeate NaCl has to stay at or below {v}.",
                                   "Product water must be under {v} NaCl.",
                                   "The permeate salinity limit is {v}."],
                                  ["Our quality cap on permeate NaCl is {v}."]),
    ("performance.water_recovery_pct", "<="): (["Recovery can't exceed {v}.",
                                                "We don't run recovery above {v}.",
                                                "The recovery limit is {v}."],
                                               ["Keep recovery at {v} or less."]),
    ("flux.water_LMH", "<="): (["Flux must stay at or below {v}.",
                                "The membrane warranty caps flux at {v}.",
                                "Don't go past {v} of flux."], ["Average flux is limited to {v}."]),
    ("performance.salt_rejection_pct", ">="): (["Salt rejection needs to be at least {v}.",
                                                "We require {v} rejection or better."],
                                               ["Rejection must not fall under {v}."]),
    ("costing.specific_energy_kWh_m3", "<="): (["Specific energy has to stay at or below {v}.",
                                                "The energy budget is {v}.",
                                                "We can't exceed {v} of specific energy."],
                                               ["Our SEC ceiling is {v}."]),
    ("costing.LCOW_usd_m3", "<="): (["The levelized cost of water must be {v} or lower.",
                                     "Water cost can be at most {v}."],
                                    ["The tariff only works if LCOW stays at or under {v}."]),
    ("performance.product_flow_m3_s", ">="): (["The plant must deliver at least {v} of product water.",
                                               "Contract output is {v} minimum.",
                                               "We need {v} of product water or more."],
                                              ["We're committed to supplying no less than {v}."]),
    ("performance.system_recovery_pct", ">="): (["Overall plant recovery has to be at least {v}.",
                                                 "System recovery can't fall below {v}."],
                                                ["We need plant recovery of {v} or better."]),
}
#: 06: one more sentence per limit, drawn by train questions only -- other words for the
#: direction ("must not ... under" / "... over"). dev and test draw from the two or three above exactly as 05 did; holdout_shift keeps
#: its one unseen sentence, which no train question uses (``check`` counts the sentences per split).
CONSTRAINTS_TRAIN_EXTRA = {
    ("permeate.flow_kg_s", ">="): ["Permeate must not come in under {v}."],
    ("permeate.nacl_ppm", "<="): ["Permeate NaCl must not go over {v}."],
    ("performance.water_recovery_pct", "<="): ["Recovery must not go over {v}."],
    ("flux.water_LMH", "<="): ["Flux can't be higher than {v}."],
    ("performance.salt_rejection_pct", ">="): ["Salt rejection can't drop below {v}."],
    ("costing.specific_energy_kWh_m3", "<="): ["Specific energy must not go over {v}."],
    ("costing.LCOW_usd_m3", "<="): ["LCOW must not go over {v}."],
    ("performance.product_flow_m3_s", ">="): ["Product water must not come in under {v}."],
    ("performance.system_recovery_pct", ">="): ["System recovery must not drop under {v}."],
}
QUESTIONS = {
    "ro_check": (["Does it meet all of these limits, and if not, which ones fail?",
                  "Are we within every limit? Tell me which ones we break, if any.",
                  "Is this point compliant with all the limits above?", "Does this operating point pass?"],
                 ["Which of these limits, if any, would be violated?"]),
    ("feed_pressure_bar", "lowest"): (["What's the lowest setting that meets all the limits? If none does, say so.",
                                       "Which is the lowest pressure that satisfies everything, if any?",
                                       "How low can we set the pump and still meet every limit? If no setting works, tell me."],
                                      ["Give me the minimum setpoint that keeps us compliant, or tell me it can't be done."]),
    ("feed_pressure_bar", "highest"): (["What's the highest setting we can run without breaking any limit? If none works, say so.",
                                        "How far up can we push the pressure and still meet every limit, if at all?",
                                        "Which is the highest pressure that still satisfies everything?"],
                                       ["What's the top setpoint that keeps us inside every limit, if any?"]),
    ("membrane_area_m2", "lowest"): (["What's the smallest area that meets all the limits? If none does, say so.",
                                      "Which is the least membrane area that satisfies everything, if any?"],
                                     ["How little membrane can we get away with and stay within every limit? Say so if none works."]),
    ("membrane_area_m2", "highest"): (["What's the largest area we can install without breaking a limit? If none works, say so.",
                                       "Which is the biggest area that still meets every limit, if any?"],
                                      ["How much membrane can we put in before a limit breaks? Tell me if no option works."]),
    "ro_membrane": (["Which is the cheapest one that meets all the limits? If none does, say so.",
                     "Pick the least expensive membrane that keeps us within every limit, or tell me none qualifies."],
                    ["Which of them is the lowest-cost element that passes everything, if any?"]),
    "swro_check": (["Does the plant meet these targets, and which ones does it miss if not?",
                    "Are we hitting every target? Tell me which ones we miss, if any."],
                   ["Which of these targets, if any, does the plant fail?"]),
    ("swro_option", "costing.specific_energy_kWh_m3"): (
        ["Which option has the lowest specific energy while meeting the targets? If none meets them, say so.",
         "Of the options that meet the targets, which uses the least energy per cubic metre?"],
        ["Which configuration meets the targets at the lowest SEC, if any does?"]),
    ("swro_option", "costing.LCOW_usd_m3"): (
        ["Which option gives the lowest levelized cost of water while meeting the targets? If none meets them, say so.",
         "Of the options that meet the targets, which one makes the cheapest water?"],
        ["Which configuration meets the targets at the lowest LCOW, if any does?"]),
}
CANDIDATES = {
    ("feed_pressure_bar", "range"): (["The pump can be set from {a} to {b} in steps of {s}.",
                                      "Allowed setpoints run from {a} to {b} in {s} steps."],
                                     ["The VFD goes from {a} to {b} in increments of {s}."]),
    ("feed_pressure_bar", "list"): (["We can choose among {list}.", "The candidate setpoints are {list}."],
                                    ["The options on the table are {list}."]),
    ("membrane_area_m2", "range"): (["We could install anywhere from {a} to {b} in steps of {s}.",
                                     "Area options run from {a} to {b} in {s} increments."],
                                    ["The rack allows {a} to {b} in steps of {s}."]),
    ("membrane_area_m2", "list"): (["The area options are {list}.", "We're considering {list}."],
                                   ["The layouts on the table give {list}."]),
}
MEMBRANE_LIST = (["The offers, cheapest first: {items}.", "In order of price, lowest first: {items}."],
                 ["Ranked by cost, cheapest first: {items}."])
VALUES_REQUEST = (["Also give me the {m}{where}.", "Please include the {m}{where}.",
                   "I'd also like the {m}{where}."], ["And what {m} would we see{where}?"])
DISTRACTORS = (["SDI15 has been around {sdi:.1f}.", "Feed pH is {ph:.1f}.", "Turbidity is under {ntu:.1f} NTU.",
                "Antiscalant dosing is {asc:.1f} mg/L.", "The cartridge filters were changed {days} days ago."],
               ["Chlorine residual is zero after the SBS dose.", "Permeate boron was {boron:.1f} mg/L last week."])

RANGES = {
    "base": {"ro": {"flow": (1.5, 6.0), "sal": (0.028, 0.042), "temp": (15, 35), "area": (150, 600),
                    "A": (3.0e-12, 5.5e-12), "B": (2.0e-8, 5.0e-8), "p": (45, 70)},
             "swro": {"flow": (0.10, 0.40), "tds": (30, 42), "temp": (15, 32), "p": (60, 80),
                      "p1_eff": (0.75, 0.88), "pxr": (0.88, 0.97), "pat": (0.75, 0.92)}},
    # holdout_shift: brinier, colder feed and bigger trains / smaller plants, unseen phrasings.
    "shift": {"ro": {"flow": (6.0, 8.0), "sal": (0.041, 0.046), "temp": (8, 14), "area": (500, 800),
                     "A": (3.0e-12, 5.5e-12), "B": (2.0e-8, 5.0e-8), "p": (60, 80)},
              "swro": {"flow": (0.05, 0.10), "tds": (42, 45), "temp": (10, 15), "p": (68, 82),
                       "p1_eff": (0.75, 0.88), "pxr": (0.88, 0.97), "pat": (0.75, 0.92)}},
}
FAMILIES = ("ro_check", "ro_scan_pressure", "ro_scan_area", "ro_membrane", "swro_check", "swro_option")
SPLITS = {"train": (1200, 101), "dev": (240, 202), "test": (240, 303), "holdout_shift": (120, 404)}
LEVEL_MIX = (0.40, 0.35, 0.25)


# ---------------------------------------------------------------------------------------------
# Tool host: the deployed MCP argument handling, minus the rate limiter.
# ---------------------------------------------------------------------------------------------

class ToolHost:
    """Runs a tool call exactly as mcp_watertap/server.py serves it.

    FastMCP validates arguments with the tool's pydantic model (unknown keys are dropped, numeric
    strings coerced), calls the function and wraps any exception as "Error executing tool ...".
    The server's own function is used, unwrapped from the rate limiter. server.py imports
    reaktoro_model, which is not installed here; a stub stands in for it -- neither exposed tool
    touches it.
    """

    def __init__(self) -> None:
        import logging
        import types
        logging.disable(logging.CRITICAL)
        if "reaktoro_model" not in sys.modules:
            stub = types.ModuleType("reaktoro_model")
            stub.ReaktoroSimulationError = type("ReaktoroSimulationError", (RuntimeError,), {})
            stub.MOLAR_MASS, stub.DEFAULT_COMPOSITION, stub.DEFAULT_MINERALS = {}, {}, ()
            stub.available_minerals = lambda: []
            stub.equilibrate = stub.composition_salinity = None
            sys.modules["reaktoro_model"] = stub
        sys.path.insert(0, str(MCP_DIR))
        import server  # noqa: E402
        self.server = server
        self.tools = {name: server.mcp._tool_manager.get_tool(name) for name in TOOLS}

    def call(self, name: str, arguments: Any) -> str:
        if name == CONVERT:
            a = arguments if isinstance(arguments, dict) else {}
            try:
                return convert_units(a.get("value"), str(a.get("from_unit", "")), str(a.get("to_unit", "")))
            except Exception as exc:  # noqa: BLE001
                return f"Error executing tool {name}: {exc}"
        if name == COMPARE:
            a = arguments if isinstance(arguments, dict) else {}
            try:
                return check_limits(a.get("limits"), a.get("values"))
            except Exception as exc:  # noqa: BLE001
                return f"Error executing tool {name}: {exc}"
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            meta = tool.fn_metadata
            parsed = meta.arg_model.model_validate(meta.pre_parse_json(arguments or {}))
            return tool.fn.__wrapped__(**parsed.model_dump_one_level())
        except Exception as exc:  # noqa: BLE001 -- mirrors Tool.run
            return f"Error executing tool {name}: {exc}"


def result_is_error(text: str | None) -> bool:
    if not text:
        return True
    if text.startswith("Error executing tool") or text.startswith("Unknown tool"):
        return True
    try:
        return "error" in json.loads(text)
    except ValueError:
        return True


def metric(result: dict, path: str) -> float:
    node: Any = result
    for part in path.split("."):
        node = node[part]
    return float(node)


# ---------------------------------------------------------------------------------------------
# Scenario generation.
# ---------------------------------------------------------------------------------------------

class Retry(Exception):
    """The sampled scenario cannot carry the target answer; draw another."""


class Gen:
    def __init__(self, rng: random.Random, split: str, level: int, host: ToolHost, u: float = 0.5):
        self.rng, self.split, self.level, self.host = rng, split, level, host
        #: One uniform draw per case, fixed across attempts, that picks the target answer. Redrawing
        #: it on every retry would favour whichever answers are easiest to fit.
        self.u = u
        self.hold = split == "holdout_shift"
        self.R = RANGES["shift" if self.hold else "base"]
        self.levels: list[int] = []
        self.units: dict[str, dict] = {}  # input key -> unit its value was stated in
        self.inputs: dict[str, dict] = {}  # input key -> {"q", "user_value", "unit"} for the shared inputs
        self.last_input: dict | None = None
        self.n_solves = 0

    def target(self, k: int, p_none: float) -> int | None:
        """The target option: none with probability p_none, else a position spread evenly over k."""
        return None if self.u < p_none else min(k - 1, int((self.u - p_none) / (1 - p_none) * k))

    def pick(self, bank: tuple[list[str], list[str]]) -> str:
        return self.rng.choice(bank[1] if self.hold else bank[0])

    def unit(self, options: list[dict]) -> dict:
        u = self.rng.choice([u for u in options if u["level"] <= self.level])
        self.levels.append(u["level"])
        return u

    def quantity(self, q: str, lo: float, hi: float, u: dict | None = None,
                 key: str | None = None) -> tuple[float, str, dict]:
        u = u or self.unit(INPUT_UNITS[q])
        user = round_grid(to_user(u, self.rng.uniform(lo, hi)), u["grid"])
        self.last_input = {"q": q, "user_value": user, "unit": u["name"]}
        if key:
            self.inputs[key] = self.last_input
        return to_tool(u, user), u["fmt"].format(user), u

    def solve(self, tool: str, args: dict) -> dict:
        self.n_solves += 1
        text = self.host.call(tool, args)
        if result_is_error(text):
            raise Retry(f"{tool} failed: {text[:120]}")
        result = json.loads(text)
        return {"args": args, "result": text, "metrics": {m: metric(result, m) for m in METRICS[tool]}}

    # -- thresholds ----------------------------------------------------------------------------

    def threshold_options(self, path: str, op: str, values: list[float]) -> list[dict]:
        """Every margin-respecting way one threshold can split these values."""
        m, spread = MARGIN[path], SPREAD[path]
        vs = sorted(set(values))
        out = []
        for i in range(len(vs) + 1):
            below = vs[i - 1] if i > 0 else None
            above = vs[i] if i < len(vs) else None
            lo = below * (1 + m) if below is not None else above * (1 - spread)
            hi = above * (1 - m) if above is not None else below * (1 + spread)
            if path in PCT_METRICS:
                hi = min(hi, 99.9)
            if lo > hi:
                continue
            # The threshold sits strictly between `below` and `above`.
            if op == ">=":
                passing = [above is not None and v >= above for v in values]
            else:
                passing = [below is not None and v <= below for v in values]
            out.append({"lo": lo, "hi": hi, "pass": passing})
        return out

    def fit_thresholds(self, cons: list[tuple[str, str]], points: list[dict], accept) -> list[dict]:
        """Choose thresholds so the points' pass/fail pattern satisfies ``accept``; render them."""
        options = [self.threshold_options(p, op, [pt["metrics"][p] for pt in points]) for p, op in cons]
        combos = [[]]
        for opts in options:
            combos = [c + [o] for c in combos for o in opts]
        good = [c for c in combos if accept([[o["pass"][j] for o in c] for j in range(len(points))])]
        if not good:
            raise Retry("no threshold design gives the target answer")
        chosen = self.rng.choice(good)
        rendered = []
        for (path, op), o in zip(cons, chosen):
            u = self.unit(METRIC_UNITS[path])
            nice = nice_in(to_user(u, o["lo"]), to_user(u, o["hi"]), self.rng)
            if nice is None:
                raise Retry("no round threshold fits")
            user, step = nice
            tool_t = to_tool(u, user)
            for pt in points:  # the rendered, rounded threshold must keep every margin
                v = pt["metrics"][path]
                if abs(v - tool_t) < MARGIN[path] * abs(v) * 0.999:
                    raise Retry("rounded threshold broke a margin")
            text = u["fmt"].format(fmt_step(user, step))
            rendered.append({"metric": path, "op": op, "threshold": tool_t, "user_value": user,
                             "user_unit": u["name"], "text": text})
        return rendered

    @staticmethod
    def passes(con: dict, v: float) -> bool:
        return v >= con["threshold"] if con["op"] == ">=" else v <= con["threshold"]

    def violations(self, cons: list[dict], pt: dict) -> list[str]:
        return sorted(c["metric"] for c in cons if not self.passes(c, pt["metrics"][c["metric"]]))

    # -- shared rendering ------------------------------------------------------------------------

    def fact(self, key: str, value_text: str) -> str:
        return self.pick(FACTS[key]).format(v=value_text)

    def constraint_sentences(self, cons: list[dict]) -> list[str]:
        # 06: a train question draws from 05's sentences and the extra ones; dev and test draw from
        # 05's with the same call as 05 (same stream, same prompts); holdout_shift from its unseen one.
        out = []
        for c in cons:
            key = (c["metric"], c["op"])
            bank = CONSTRAINTS[key][0] + CONSTRAINTS_TRAIN_EXTRA[key] if self.split == "train" else None
            out.append((self.rng.choice(bank) if bank else self.pick(CONSTRAINTS[key])).format(v=c["text"]))
        return out

    def values_request(self, paths: list[str], where: str) -> str:
        names = [self.rng.choice(METRIC_NAMES[p]) for p in paths]
        m = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return self.pick(VALUES_REQUEST).format(m=m, where=where)

    def distractor(self) -> str:
        r = self.rng
        return self.pick(DISTRACTORS).format(sdi=r.uniform(1.5, 4.5), ph=r.uniform(7.6, 8.3),
                                             ntu=r.uniform(0.1, 0.9), asc=r.uniform(1.0, 4.0),
                                             days=r.randint(2, 30), boron=r.uniform(0.6, 1.4))

    def assemble(self, opener: str, facts: list[str], cons: list[str], extra: list[str],
                 question: str, values: str | None) -> str:
        r = self.rng
        r.shuffle(facts)
        r.shuffle(cons)
        if r.random() < 0.25:
            facts.insert(r.randrange(len(facts) + 1), self.distractor())
        ask = " ".join(extra + [question] + ([values] if values else []))
        body = " ".join(facts)
        limits = " ".join(cons)
        if r.random() < 0.5:
            return f"{opener} {body} {limits} {ask}"
        return f"{opener} {body}\n\n{limits} {ask}"

    def pick_values(self, tool: str, cons: list[dict]) -> list[str]:
        constrained = [c["metric"] for c in cons]
        n = self.rng.choice([1, 1, 2])
        pool = list(METRICS[tool])
        first = self.rng.choice(constrained) if constrained and self.rng.random() < 0.6 else self.rng.choice(pool)
        rest = [p for p in pool if p != first]
        return [first] + self.rng.sample(rest, n - 1)

    # -- RO scenario pieces ----------------------------------------------------------------------

    def ro_base(self, skip: tuple[str, ...] = ()) -> tuple[dict, dict]:
        R, r = self.R["ro"], self.rng
        stated, texts = {}, {}
        for key, q, rng_key in (("feed_flow_mass_kg_s", "ro_flow", "flow"),
                                ("feed_nacl_mass_frac", "salinity", "sal"),
                                ("membrane_area_m2", "area", "area"),
                                ("feed_pressure_bar", "pressure", "p")):
            if key not in skip:
                stated[key], texts[key], self.units[key] = self.quantity(q, *R[rng_key], key=key)
        if r.random() < 0.85:
            key = "feed_temperature_c"
            stated[key], texts[key], self.units[key] = self.quantity("temp", *R["temp"], key=key)
        if "A_comp" not in skip and r.random() < 0.75:
            stated["A_comp"], a_txt, _ = self.quantity("A", *R["A"], key="A_comp")
            stated["B_comp"], b_txt, _ = self.quantity("B", *R["B"], key="B_comp")
            texts["AB"] = (a_txt, b_txt)
        return stated, texts

    def ro_fact_sentences(self, texts: dict) -> list[str]:
        out = []
        for key, txt in texts.items():
            if key == "AB":
                out.append(self.pick(FACTS["AB"]).format(A=txt[0], B=txt[1]))
            else:
                out.append(self.fact(key, txt))
        return out

    def ro_constraint_pool(self) -> list[tuple[str, str]]:
        return [("permeate.flow_kg_s", ">="), ("permeate.nacl_ppm", "<="),
                ("performance.water_recovery_pct", "<="), ("flux.water_LMH", "<="),
                ("performance.salt_rejection_pct", ">=")]


def check_family(g: Gen, tool: str) -> dict:
    """ro_check / swro_check: one operating point, pass or fail, which limits break."""
    r = g.rng
    if tool == RO:
        stated, texts = g.ro_base()
        extra = []
        story = r.random()
        if story < 0.35:  # a disturbance: the old value is a distractor, the new one is stated
            key = r.choice([k for k in DISTURB_NAMES if k in texts])
            u = g.units[key]
            old_tool = stated[key] * r.choice([0.85, 0.9, 0.93, 1.07, 1.1, 1.15])
            old_txt = u["fmt"].format(round_grid(to_user(u, old_tool), u["grid"]))
            sentence = g.pick(DISTURBANCE).format(name=DISTURB_NAMES[key], lname=DISTURB_NAMES[key].lower(),
                                                  old=old_txt, new=texts[key], event=r.choice(EVENTS[key]))
            del texts[key]
            facts = g.ro_fact_sentences(texts) + [sentence]
        elif story < 0.6:  # a proposed setpoint
            p_txt = texts.pop("feed_pressure_bar")
            facts = g.ro_fact_sentences(texts) + [g.pick(PROPOSAL).format(v=p_txt)]
        else:
            facts = g.ro_fact_sentences(texts)
        pt = g.solve(RO, stated)
        pool = g.ro_constraint_pool()
        opener, question = g.pick(OPENERS["ro_check"]), g.pick(QUESTIONS["ro_check"])
        family = "ro_check"
    else:
        stated, facts = swro_base(g)
        pt = g.solve(SWRO, stated)
        pool = [("costing.specific_energy_kWh_m3", "<="), ("costing.LCOW_usd_m3", "<="),
                ("performance.product_flow_m3_s", ">="), ("performance.system_recovery_pct", ">=")]
        opener, question = g.pick(OPENERS["swro_check"]), g.pick(QUESTIONS["swro_check"])
        family, extra = "swro_check", []
    cons = r.sample(pool, r.choice([2, 3, 3, 4] if tool == RO else [2, 3, 3]))
    want_pass = g.u < 0.5
    rendered = g.fit_thresholds(cons, [pt], lambda m: all(m[0]) == want_pass)
    decision = "pass" if want_pass else "fail"
    vals = g.pick_values(tool, rendered)
    prompt = g.assemble(opener, facts, g.constraint_sentences(rendered), extra, question,
                        g.values_request(vals, r.choice(["", " at this point"])))
    pt["key"], pt["label"] = "p0", "point"
    return {"family": family, "tool": tool, "prompt": prompt, "points": [pt], "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "passfail",
            "gold": {"decision": decision, "violations": g.violations(rendered, pt),
                     "values": {p: pt["metrics"][p] for p in vals}},
            "proof": ["p0"]}


def swro_base(g: Gen, skip: tuple[str, ...] = ()) -> tuple[dict, list[str]]:
    R, r = g.R["swro"], g.rng
    stated, facts = {}, []
    stated["feed_flow_m3_s"], t, _ = g.quantity("swro_flow", *R["flow"], key="feed_flow_m3_s")
    facts.append(g.fact("feed_flow_m3_s", t))
    stated["feed_tds_g_L"], t, _ = g.quantity("tds", *R["tds"], key="feed_tds_g_L")
    facts.append(g.fact("feed_tds_g_L", t))
    if r.random() < 0.8:
        stated["feed_temperature_c"], t, _ = g.quantity("temp", *R["temp"], key="feed_temperature_c")
        facts.append(g.pick(FACTS["swro_temp"]).format(v=t))
    if "p1_pressure_bar" not in skip:
        stated["p1_pressure_bar"], t, _ = g.quantity("pressure", *R["p"], key="p1_pressure_bar")
        facts.append(g.fact("p1_pressure_bar", t))
    if "p1_efficiency" not in skip and r.random() < 0.6:
        stated["p1_efficiency"], t, _ = g.quantity("eff", *R["p1_eff"], key="p1_efficiency")
        facts.append(g.fact("p1_efficiency", t))
    if "erd" not in skip and r.random() < 0.7:
        args, text = erd_config(g, r.random() < 0.7)
        stated.update(args)
        facts.append(text)
    return stated, facts


def erd_config(g: Gen, px: bool) -> tuple[dict, str]:
    R = g.R["swro"]
    if px:
        eff, t, _ = g.quantity("eff", *R["pxr"], key="pxr_efficiency")
        return {"erd_type": "pressure_exchanger", "pxr_efficiency": eff}, g.pick(FACTS["erd_px"]).format(v=t)
    eff, t, _ = g.quantity("eff", *R["pat"], key="erd_efficiency")
    return {"erd_type": "pump_as_turbine", "erd_efficiency": eff}, g.pick(FACTS["erd_pat"]).format(v=t)


def scan_family(g: Gen, param: str) -> dict:
    """ro_scan_pressure / ro_scan_area: an ordered set of settings, lowest or highest that passes."""
    r = g.rng
    stated, texts = g.ro_base(skip=(param,))
    q = "pressure" if param == "feed_pressure_bar" else "area"
    u = g.unit(INPUT_UNITS[q])
    step = r.choice(SCAN_STEPS[(q, u["name"])])
    k = r.choice([3, 4, 4, 5])
    lo, hi = (g.R["ro"]["p"] if q == "pressure" else g.R["ro"]["area"])
    span_tool = to_tool(u, to_user(u, lo) + (k - 1) * step) - lo
    start_tool = r.uniform(lo, max(lo, hi - span_tool))
    start = round_grid(to_user(u, start_tool), step)
    users = [round(start + j * step, 10) for j in range(k)]
    points = []
    for j, uv in enumerate(users):
        pt = g.solve(RO, {**stated, param: to_tool(u, uv)})
        pt["key"], pt["label"], pt["user_value"] = f"p{j}", u["fmt"].format(uv), uv
        pt["inputs"] = {param: {"q": q, "user_value": uv, "unit": u["name"]}}
        points.append(pt)
    objective = r.choice(["lowest", "highest"])
    # Constraints must be monotone along the scan; classify each as failing at the low or high end.
    monotone = []
    for path, op in g.ro_constraint_pool():
        vs = [p["metrics"][path] for p in points]
        diffs = [b - a for a, b in zip(vs, vs[1:])]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            inc = diffs[0] > 0
            monotone.append((path, op, "low" if (op == ">=") == inc else "high"))
    need = "low" if objective == "lowest" else "high"
    if not any(side == need for *_, side in monotone):
        raise Retry("no constraint binds on the needed side")
    binding = r.choice([c for c in monotone if c[2] == need])
    others = [c for c in monotone if c != binding]
    chosen = [binding] + r.sample(others, min(len(others), r.choice([1, 1, 2])))
    sides = {path: side for path, _, side in chosen}
    target = g.target(k, 0.2)

    def decide(mask):
        passing = [j for j, row in enumerate(mask) if all(row)]
        if not passing:
            return None
        return passing[0] if objective == "lowest" else passing[-1]

    cons = [(p, op) for p, op, _ in chosen]
    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    status = []
    for pt in points:
        failed = {sides[c["metric"]] for c in rendered if not g.passes(c, pt["metrics"][c["metric"]])}
        status.append(failed)
    decision, proof = scan_proof(status, objective)
    assert decision == target
    extra = [g.pick(CANDIDATES[(param, "range")]).format(a=u["fmt"].format(users[0]), b=u["fmt"].format(users[-1]),
                                                         s=u["fmt"].format(step))
             if r.random() < 0.5 else
             g.pick(CANDIDATES[(param, "list")]).format(list=_or_list([p["label"] for p in points]))]
    family = "ro_scan_pressure" if q == "pressure" else "ro_scan_area"
    vals = g.pick_values(RO, rendered)  # asked for either way; graded only when a setting works
    prompt = g.assemble(g.pick(OPENERS[family]), g.ro_fact_sentences(texts), g.constraint_sentences(rendered),
                        extra, g.pick(QUESTIONS[(param, objective)]),
                        g.values_request(vals, r.choice([" at that setting", " for the one you pick", ""])))
    values = {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}
    return {"family": family, "tool": RO, "prompt": prompt, "inputs": g.inputs,
            "points": points, "constraints": rendered, "decision_kind": "number", "objective": objective,
            "scan": {"param": param, "unit": u["name"], "users": users},
            "gold": {"decision": None if decision is None else users[decision],
                     "decision_tool": None if decision is None else to_tool(u, users[decision]),
                     "violations": [], "values": values},
            "proof": [f"p{j}" for j in proof]}


def scan_proof(status: list[set], objective: str) -> tuple[int | None, list[int]]:
    """The answer, and the fewest points whose simulation proves it, given monotone metrics."""
    k = len(status)
    lows = [j for j, s in enumerate(status) if "low" in s]
    highs = [j for j, s in enumerate(status) if "high" in s]
    if lows and lows != list(range(len(lows))):
        raise Retry("low-side failures are not a prefix")
    if highs and highs != list(range(k - len(highs), k)):
        raise Retry("high-side failures are not a suffix")
    passing = [j for j, s in enumerate(status) if not s]
    if passing:
        if objective == "lowest":
            i = passing[0]
            return i, ([i - 1, i] if i > 0 else [i])
        i = passing[-1]
        return i, ([i, i + 1] if i < k - 1 else [i])
    both = [j for j, s in enumerate(status) if s == {"low", "high"}]
    if both:
        return None, [both[0]]
    if not highs:
        return None, [k - 1]
    if not lows:
        return None, [0]
    return None, [lows[-1], highs[0]]


def _or_list(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " or " + items[-1]


def membrane_family(g: Gen) -> dict:
    """ro_membrane: candidate (A, B) pairs in price order; the cheapest that passes, or none."""
    r = g.rng
    stated, texts = g.ro_base(skip=("A_comp",))
    k = r.choice([2, 3, 3, 4])
    ua, ub = g.unit(INPUT_UNITS["A"]), g.unit(INPUT_UNITS["B"])
    points, items, seen = [], [], set()
    for j in range(k):
        a, a_txt, _ = g.quantity("A", 2.8e-12, 6.0e-12, ua)
        a_in = g.last_input
        b, b_txt, _ = g.quantity("B", 1.5e-8, 6.0e-8, ub)
        b_in = g.last_input
        if (a_txt, b_txt) in seen:
            raise Retry("duplicate membrane")
        seen.add((a_txt, b_txt))
        pt = g.solve(RO, {**stated, "A_comp": a, "B_comp": b})
        label = "ABCD"[j]
        pt["key"], pt["label"] = f"p{j}", label
        pt["inputs"] = {"A_comp": a_in, "B_comp": b_in}
        points.append(pt)
        items.append(f"Membrane {label} (A = {a_txt}, B = {b_txt})")
    pool = g.ro_constraint_pool()
    cons = [("permeate.flow_kg_s", ">="), ("permeate.nacl_ppm", "<=")] + r.sample(pool[2:], r.choice([0, 0, 1]))
    target = g.target(k, 0.2)

    def decide(mask):
        return next((j for j, row in enumerate(mask) if all(row)), None)

    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    decision = target
    proof = list(range(k)) if decision is None else list(range(decision + 1))
    vals = g.pick_values(RO, rendered)
    prompt = g.assemble(g.pick(OPENERS["ro_membrane"]), g.ro_fact_sentences(texts), g.constraint_sentences(rendered),
                        [g.pick(MEMBRANE_LIST).format(items="; ".join(items))], g.pick(QUESTIONS["ro_membrane"]),
                        g.values_request(vals, r.choice([" for the one you pick", ""])))
    return {"family": "ro_membrane", "tool": RO, "prompt": prompt, "points": points, "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "label",
            "gold": {"decision": None if decision is None else "ABCD"[decision], "violations": [],
                     "values": {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}},
            "proof": [f"p{j}" for j in proof]}


def option_family(g: Gen) -> dict:
    """swro_option: plant configurations; the lowest SEC or LCOW among those meeting the targets."""
    r = g.rng
    dims = r.sample(["p1_pressure_bar", "erd", "p1_efficiency"], r.choice([1, 2, 2]))
    stated, facts = swro_base(g, skip=tuple(dims))
    k = r.choice([2, 3, 3])
    R = g.R["swro"]
    u_p = g.unit(INPUT_UNITS["pressure"]) if "p1_pressure_bar" in dims else None
    u_e = g.unit(INPUT_UNITS["eff"])
    points, items = [], []
    for j in range(k):
        args, pieces, inputs = dict(stated), [], {}
        if "p1_pressure_bar" in dims:
            args["p1_pressure_bar"], t, _ = g.quantity("pressure", *R["p"], u_p)
            inputs["p1_pressure_bar"] = g.last_input
            pieces.append(f"the pump at {t}")
        if "p1_efficiency" in dims:
            args["p1_efficiency"], t, _ = g.quantity("eff", *R["p1_eff"], u_e)
            inputs["p1_efficiency"] = g.last_input
            pieces.append(f"a pump efficiency of {t}")
        if "erd" in dims:
            px = r.random() < 0.6
            eff, t, _ = g.quantity("eff", *(R["pxr"] if px else R["pat"]), u_e)
            if px:
                args.update(erd_type="pressure_exchanger", pxr_efficiency=eff)
                inputs["pxr_efficiency"] = g.last_input
                pieces.append(f"a pressure exchanger at {t} efficiency")
            else:
                args.update(erd_type="pump_as_turbine", erd_efficiency=eff)
                inputs["erd_efficiency"] = g.last_input
                pieces.append(f"a pump-as-turbine at {t} efficiency")
        label = "ABC"[j]
        items.append(f"Option {label}: " + ", ".join(pieces) + ".")
        pt = g.solve(SWRO, args)
        pt["key"], pt["label"] = f"p{j}", label
        pt["inputs"] = inputs
        points.append(pt)
    if len({tuple(sorted((k2, str(v)) for k2, v in p["args"].items())) for p in points}) < k:
        raise Retry("duplicate options")
    objective = r.choice(["costing.specific_energy_kWh_m3", "costing.LCOW_usd_m3"])
    pool = [("performance.product_flow_m3_s", ">="), ("costing.LCOW_usd_m3", "<="),
            ("costing.specific_energy_kWh_m3", "<="), ("performance.system_recovery_pct", ">=")]
    cons = [("performance.product_flow_m3_s", ">=")] + r.sample(
        [c for c in pool[1:] if c[0] != objective], r.choice([0, 1]))
    target = g.target(k, 0.15)
    obj = [p["metrics"][objective] for p in points]

    def decide(mask):
        passing = sorted((obj[j], j) for j, row in enumerate(mask) if all(row))
        if not passing:
            return None
        if len(passing) > 1 and passing[1][0] < passing[0][0] * 1.02:
            return "tie"
        return passing[0][1]

    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    decision = target
    vals = g.pick_values(SWRO, rendered)
    prompt = g.assemble(g.pick(OPENERS["swro_option"]), facts, g.constraint_sentences(rendered),
                        [" ".join(items)], g.pick(QUESTIONS[("swro_option", objective)]),
                        g.values_request(vals, r.choice([" for the option you pick", ""])))
    return {"family": "swro_option", "tool": SWRO, "prompt": prompt, "points": points, "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "label", "objective": objective,
            "gold": {"decision": None if decision is None else "ABC"[decision], "violations": [],
                     "values": {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}},
            "proof": [f"p{j}" for j in range(k)]}


FAMILY_FNS = {
    "ro_check": lambda g: check_family(g, RO),
    "ro_scan_pressure": lambda g: scan_family(g, "feed_pressure_bar"),
    "ro_scan_area": lambda g: scan_family(g, "membrane_area_m2"),
    "ro_membrane": membrane_family,
    "swro_check": lambda g: check_family(g, SWRO),
    "swro_option": option_family,
}

_HOST: ToolHost | None = None


def _host() -> ToolHost:
    global _HOST
    if _HOST is None:
        _HOST = ToolHost()
    return _HOST


def make_case(job: tuple[str, int, int]) -> dict:
    split, seed, idx = job
    family = FAMILIES[idx % len(FAMILIES)]
    rng = random.Random(f"{seed}:{idx}")
    level = rng.choices((0, 1, 2), LEVEL_MIX)[0]
    u = random.Random(f"{seed}:{idx}:target").random()
    solves, reasons = 0, Counter()
    for attempt in range(200):
        g = Gen(rng, split, level, _host(), u)
        try:
            case = FAMILY_FNS[family](g)
        except Retry as exc:
            solves += g.n_solves
            reasons[str(exc).split(":")[0]] += 1
            continue
        solves += g.n_solves
        case.update(id=f"{family}-{split}-{idx:05d}", split=split, level=max(g.levels, default=0),
                    optimal_calls=len(case["proof"]), attempts=attempt + 1, solves=solves)
        return case
    raise RuntimeError(f"{split}/{idx} ({family}): no case after 200 attempts: {dict(reasons)}")


# ---------------------------------------------------------------------------------------------
# Reward. Pure Python: runs in the training venv without WaterTAP.
# ---------------------------------------------------------------------------------------------

WEIGHTS = {"calls": 0.25, "decision": 0.35, "violations": 0.10, "values": 0.20, "format": 0.05,
           "efficiency": 0.05}
ARG_TOL = 0.005      # relative tolerance on each tool argument
VALUE_TOL = 0.015    # relative tolerance on each reported number (arguments off by ARG_TOL move outputs <= 1.43%)
UNGATED = 0.3        # share of the decision/violations credit paid without the simulations that prove it
NONE_WORDS = {"none", "null", "n/a", "no option", "not feasible", "infeasible", "nothing", "no"}


def parse_number(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        m = re.search(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?", x)
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                return None
    return None


def _same_arg(a: Any, b: Any) -> bool:
    if b is None or isinstance(b, str):
        return (str(a).strip().lower() if a is not None else None) == (b.lower() if b else None)
    x = parse_number(a)
    return x is not None and abs(x - b) <= ARG_TOL * abs(b)


def call_fidelity(case: dict, call: dict) -> tuple[float, str | None]:
    """How faithfully one call reproduces a point of the scenario: (score in [0,1], covered point).

    Scored against the closest point: the share of the scenario's stated arguments the call got
    right (after unit conversion, within ARG_TOL), minus 0.25 for each argument it set that the user
    never gave (unless it equals the tool default) and for each argument the tool does not have.
    A call covers a point when it gets every argument right, changes nothing else and ran cleanly.
    """
    name, args = call.get("name"), call.get("arguments")
    if name != case["tool"] or not isinstance(args, dict):
        return 0.0, None
    defaults = DEFAULTS[name]
    known = {k: v for k, v in args.items() if k in defaults}
    unknown = len(args) - len(known)
    ran = not result_is_error(call.get("result"))
    best, covered = 0.0, None
    for pt in case["points"]:
        exp = pt["args"]
        right = sum(_same_arg(known.get(k, defaults[k]), v) for k, v in exp.items())
        stray = sum(1 for k, v in known.items() if k not in exp and not _same_arg(v, defaults[k]))
        score = max(0.0, right / len(exp) - 0.25 * (stray + unknown))
        if right == len(exp) and stray == 0 and ran:
            covered = pt["key"]
        best = max(best, score)
    return best, covered


def json_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level {...} in text, string-aware."""
    spans, depth, start, in_str, esc = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = depth > 0
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def extract_answer(text: str) -> tuple[dict | None, tuple[int, int] | None, bool]:
    """The first JSON object with a "decision" key: (object, span, nothing-after-it)."""
    for s, e in json_spans(text or ""):
        try:
            obj = json.loads(text[s:e])
        except ValueError:
            continue
        if isinstance(obj, dict) and "decision" in obj:
            return obj, (s, e), not text[e:].strip()
    return None, None, False


def _none_like(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x.strip().lower().rstrip(".") in NONE_WORDS)


def decision_correct(case: dict, pred: Any) -> bool:
    gold, kind = case["gold"]["decision"], case["decision_kind"]
    if _none_like(pred):
        return gold is None
    if gold is None:
        return False
    if kind == "passfail":
        return isinstance(pred, str) and pred.strip().lower().rstrip(".") in (gold, gold + "s", gold + "ed")
    if kind == "number":
        x = parse_number(pred)
        if x is None:
            return False
        return (abs(x - gold) <= 0.002 * abs(gold)
                or abs(x - case["gold"]["decision_tool"]) <= ARG_TOL * abs(case["gold"]["decision_tool"]))
    m = re.match(r"\s*(?:option|membrane|configuration|config)?\s*([A-Da-d])\b", str(pred), re.I)
    return bool(m) and m.group(1).upper() == gold


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
    return out


def score(case: dict, calls: list[dict], final_text: str) -> dict:
    """Reward for one episode, with the credits a trainer places on tokens.

    calls: [{"name", "arguments" (dict, or None if unparseable), "result" (tool text)}] in order.
    final_text: the last assistant turn. Credits: ("call", i) at that call's end; ("decision",),
    ("violations",) and ("value", path) at the answer field; ("close",) at the answer's closing
    brace (format + efficiency).
    """
    comps: dict[str, float] = {}
    is_sim = [c.get("name") not in LOCAL_TOOLS for c in calls]
    n = sum(is_sim)  # simulations; convert_units and check_limits calls earn and cost nothing
    fid = [call_fidelity(case, c) if sim else (0.0, None) for c, sim in zip(calls, is_sim)]
    covered = {cov for _, cov in fid if cov}
    evidence = set(case["proof"]) <= covered
    comps["calls"] = sum(f for (f, _), sim in zip(fid, is_sim) if sim) / n if n else 0.0
    ans, _, clean_end = extract_answer(final_text)
    ans = ans or {}
    correct = bool(ans) and decision_correct(case, ans.get("decision"))
    gate = 1.0 if evidence else UNGATED
    comps["decision"] = gate * correct
    if case["decision_kind"] == "passfail":
        pred = ans.get("violations")
        pred_set = {str(x).strip() for x in pred} if isinstance(pred, list) else None
        gold_set = set(case["gold"]["violations"])
        if pred_set is None:
            v = 0.0
        elif not gold_set:
            v = float(not pred_set)
        else:
            v = 2 * len(pred_set & gold_set) / (len(pred_set) + len(gold_set))
        comps["violations"] = gate * v
    gold_vals = case["gold"]["values"]
    value_hits = {}
    if gold_vals:
        pred_vals = flatten(ans.get("values"))
        for path, gv in gold_vals.items():
            x = parse_number(pred_vals.get(path))
            value_hits[path] = x is not None and abs(x - gv) <= VALUE_TOL * abs(gv)
        comps["values"] = sum(value_hits.values()) / len(gold_vals)
    comps["format"] = float(bool(ans) and clean_end and isinstance(ans.get("violations"), list)
                            and isinstance(ans.get("values"), dict))
    comps["efficiency"] = min(1.0, case["optimal_calls"] / n) if (correct and evidence and n) else 0.0
    z = sum(WEIGHTS[k] for k in comps)
    total = sum(WEIGHTS[k] * v for k, v in comps.items()) / z
    credits = [(("call", i), WEIGHTS["calls"] * f / n / z if sim else 0.0) for i, ((f, _), sim) in enumerate(zip(fid, is_sim))]
    credits.append((("decision",), WEIGHTS["decision"] * comps["decision"] / z))
    if "violations" in comps:
        credits.append((("violations",), WEIGHTS["violations"] * comps["violations"] / z))
    for path, hit in value_hits.items():
        credits.append((("value", path), WEIGHTS["values"] * hit / len(gold_vals) / z))
    credits.append((("close",), (WEIGHTS["format"] * comps["format"] + WEIGHTS["efficiency"] * comps["efficiency"]) / z))
    return {"total": total, "components": comps, "evidence": evidence, "covered": sorted(covered),
            "n_calls": n, "n_converts": sum(c.get("name") == CONVERT for c in calls),
            "n_compares": sum(c.get("name") == COMPARE for c in calls), "decision_correct": correct, "credits": credits}


# ---------------------------------------------------------------------------------------------
# Reference episodes and null baselines.
# ---------------------------------------------------------------------------------------------

def gold_answer(case: dict) -> dict:
    g = case["gold"]
    return {"decision": g["decision"], "violations": g["violations"],
            "values": {k: float(f"{v:.4g}") for k, v in g["values"].items()}}


def oracle_calls(case: dict) -> list[dict]:
    by_key = {p["key"]: p for p in case["points"]}
    return [{"name": case["tool"], "arguments": {k: (float(f"{v:.6g}") if isinstance(v, float) else v)
                                                 for k, v in by_key[key]["args"].items()},
             "result": by_key[key]["result"]} for key in case["proof"]]


def baselines(case: dict) -> dict[str, float]:
    """Totals for the reference episode and for strategies that should not earn much."""
    gold = gold_answer(case)
    calls = oracle_calls(case)
    out = {"oracle": score(case, calls, json.dumps(gold))["total"],
           "oracle_plus_2_calls": score(case, calls + calls[:1] * 2, json.dumps(gold))["total"],
           "gold_answer_no_calls": score(case, [], json.dumps(gold))["total"]}
    wrong = dict(gold)
    kind = case["decision_kind"]
    if kind == "passfail":
        wrong["decision"] = "fail" if gold["decision"] == "pass" else "pass"
    else:
        wrong["decision"] = "Z" if kind == "label" else -1
    out["proof_calls_wrong_decision"] = score(case, calls, json.dumps(wrong))["total"]
    # No tools: the best fixed guess a policy could learn per family.
    if kind == "passfail":
        guesses = {"no_calls_say_pass": "pass", "no_calls_say_fail": "fail"}
    elif kind == "number":
        users = case["scan"]["users"]
        guesses = {"no_calls_first_option": users[0], "no_calls_last_option": users[-1], "no_calls_none": None}
    else:
        guesses = {"no_calls_first_option": "A", "no_calls_last_option": case["points"][-1]["label"],
                   "no_calls_none": None}
    for name, dec in guesses.items():
        out[name] = score(case, [], json.dumps({"decision": dec, "violations": [], "values": {}}))["total"]
    return out


# ---------------------------------------------------------------------------------------------
# Benchmark decontamination.
# ---------------------------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def read_xlsx_text(path: Path, first_sheet_only: bool = False) -> str:
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    parts = []
    sheets = list(ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS))
    for sheet in sheets[:1] if first_sheet_only else sheets:
        target = rels[sheet.get(f"{{{_NS['r']}}}id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        for c in ET.fromstring(z.read(target)).iter(f"{{{_NS['m']}}}c"):
            v = c.find("m:v", _NS)
            if c.get("t") == "s" and v is not None:
                parts.append(shared[int(v.text)])
            elif c.get("t") == "inlineStr":
                parts.append("".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t")))
            elif v is not None:
                parts.append(v.text)
    return "\n".join(parts)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())


def _numbers(text: str, sig: int = 4) -> set[str]:
    out = set()
    for m in re.findall(r"\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text):
        try:
            out.add(f"{float(m.replace(',', '')):.{sig}g}")
        except ValueError:
            continue
    return out


#: The inputs that define a scenario; a case "reuses" a benchmark scenario if all of them appear in
#: one benchmark workbook (compared at 4 significant figures, in the tool's units).
CORE_INPUTS = {RO: ("feed_flow_mass_kg_s", "feed_nacl_mass_frac", "membrane_area_m2", "feed_pressure_bar"),
               SWRO: ("feed_flow_m3_s", "feed_tds_g_L", "p1_pressure_bar")}


def decontamination(cases: list[dict], n: int = 8) -> dict:
    """Does any generated prompt share text, or a whole scenario, with the benchmark?"""
    files = sorted(BENCH_DIR.glob("D*/*.xlsx"))
    bench = {f.name: read_xlsx_text(f) for f in files}
    grams = set()
    for text in bench.values():
        w = _words(text)
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    bench_nums = {name: _numbers(text) for name, text in bench.items()}
    hits, examples, reused = [], [], []
    for c in cases:
        w = _words(c["prompt"])
        h = sum(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1))
        hits.append(h)
        if h and len(examples) < 5:
            examples.append((c["id"], [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)
                                       if tuple(w[i:i + n]) in grams][:3]))
        for p in c["points"]:
            core = {f"{p['args'][k]:.4g}" for k in CORE_INPUTS[c["tool"]] if k in p["args"]}
            match = [name for name, nums in bench_nums.items() if core <= nums]
            if match:
                reused.append((c["id"], p["key"], match[:3]))
                break
    sizes = sorted(len(v) for v in bench_nums.values())
    return {"benchmark_files": len(files), "benchmark_8grams": len(grams),
            "prompts_with_any_8gram": sum(h > 0 for h in hits), "max_8gram_hits": max(hits, default=0),
            "examples": examples, "cases_reusing_a_benchmark_scenario": len(reused),
            "reuse_examples": reused[:10],
            "numbers_per_workbook_median": sizes[len(sizes) // 2] if sizes else 0}


# ---------------------------------------------------------------------------------------------
# Rollouts: the policy in the loop with the real simulators.
#
# The training venv has torch but not WaterTAP, so the simulators run in worker processes under
# .venv-watertap (the ``worker`` subcommand: ToolHost over JSON lines), with a result cache -- the
# simulators are deterministic, so an identical call is answered from memory.
#
# The turn protocol is 03's: only <|im_end|> ends a turn (<|endoftext|> is suppressed); a turn with
# <tool_call> blocks is answered with the results in one user turn and the policy writes again; a
# turn without calls is the answer and ends the episode. Episodes are built by token concatenation,
# which equals the chat template's own rendering (``check-harness`` verifies it on reference episodes).
# ---------------------------------------------------------------------------------------------

WATERTAP_PY = REPO / ".venv-watertap" / "bin" / "python"
MODEL = "Qwen/Qwen3.5-0.8B"
TEMPLATE_KWARGS = {"enable_thinking": False}
RUNS = HERE / f"{PREFIX}_runs"
#: Simulations per episode; calls beyond this are answered with an error instead of run.
MAX_CALLS = 8
#: convert_units calls per episode (a reference episode needs at most 12).
MAX_CONVERTS = 16
#: check_limits calls per episode (a reference episode needs one).
MAX_COMPARES = 4
#: Tool rounds per episode; a turn that still calls after the last round ends the episode unanswered.
MAX_ROUNDS = 6
CALL_BUDGET_ERROR = "Error: call budget exhausted (8 simulations per question); answer with what you have."
CONVERT_BUDGET_ERROR = "Error: conversion budget exhausted (16 per question); answer with what you have."
COMPARE_BUDGET_ERROR = "Error: check budget exhausted (4 per question); answer with what you have."
MALFORMED_CALL_ERROR = "Error: malformed tool call"


class ToolPool:
    """Worker processes serving ToolHost, and a cache of every call already answered."""

    def __init__(self, n: int = 4) -> None:
        import os
        import subprocess
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        self.procs = [subprocess.Popen([str(WATERTAP_PY), str(Path(__file__).resolve()), "worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
                      for _ in range(n)]
        self.cache: dict[str, str] = {}
        self.solved = 0

    @staticmethod
    def key(name: str, arguments: Any) -> str:
        return json.dumps([name, arguments], sort_keys=True, default=str)

    def _run(self, proc, jobs: list[tuple[str, str, Any]]) -> list[tuple[str, str]]:
        out = []
        for key, name, arguments in jobs:
            proc.stdin.write(json.dumps({"name": name, "arguments": arguments}) + "\n")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("tool worker died")
            out.append((key, json.loads(line)["result"]))
        return out

    def call_many(self, calls: list[tuple[str, Any]]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        todo, seen = [], set()
        for name, arguments in calls:
            k = self.key(name, arguments)
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append((k, name, arguments))
        if todo:
            chunks = [todo[i::len(self.procs)] for i in range(len(self.procs))]
            with ThreadPoolExecutor(len(self.procs)) as ex:
                for part in ex.map(self._run, self.procs, chunks):
                    self.cache.update(part)
            self.solved += len(todo)
        return [self.cache[self.key(n, a)] for n, a in calls]

    def close(self) -> None:
        for p in self.procs:
            p.stdin.close()
            p.wait(timeout=30)


def _schema_types(name: str, param: str) -> set[str]:
    decl = next((d["function"] for d in _tool_decls() if d["function"]["name"] == name), None)
    prop = (decl or {}).get("parameters", {}).get("properties", {}).get(param, {})
    return {t.get("type") for t in prop.get("anyOf", [prop])}


_DECLS: list[dict] | None = None


def _tool_decls() -> list[dict]:
    global _DECLS
    if _DECLS is None:
        _DECLS = json.loads(TOOLS_JSON.read_text())
    return [d for d in _DECLS if COMPARE_ON or d["function"]["name"] != COMPARE]


def convert_param(name: str, param: str, raw: str) -> Any:
    """A <parameter> value as the deployed tool-call parser types it: by the tool's schema."""
    value = raw.strip()
    if value.lower() in ("null", "none"):
        return None
    types = _schema_types(name, param)
    if "number" in types or "integer" in types:
        try:
            return float(value)
        except ValueError:
            return value  # left as text; the server's validation rejects it
    if "string" in types:
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_calls(turn: str) -> list[dict]:
    """Every <tool_call> block in one assistant turn, in Qwen3.5's XML form."""
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", turn, flags=re.S):
        fn = re.fullmatch(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", body, flags=re.S)
        if fn is None:
            calls.append({"name": None, "arguments": None, "malformed": True})
            continue
        name = fn.group(1)
        params = re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fn.group(2), flags=re.S)
        calls.append({"name": name, "arguments": {k: convert_param(name, k, v) for k, v in params}})
    return calls


def call_block(name: str, arguments: dict) -> str:
    """One call exactly as the chat template writes it in an assistant turn."""
    params = "".join(f"<parameter={k}>\n{json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}\n</parameter>\n"
                     for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def turn_text(turn: dict) -> str:
    """An assistant turn of calls as the chat template writes it: the calls, one per line, after the turn's
    own line and a blank line when it has one (07: the fix in a recovery episode names what it fixes)."""
    calls = "\n".join(call_block(c["name"], c["arguments"]) for c in turn["calls"])
    return f"{turn['text']}\n\n{calls}" if turn.get("text") else calls


def render_prompt(tokenizer, case: dict) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case["prompt"]}]
    return tokenizer.apply_chat_template(msgs, tools=_tool_decls(), tokenize=False, add_generation_prompt=True,
                                         **TEMPLATE_KWARGS)


def generation_prefix(tokenizer) -> str:
    """What add_generation_prompt appends: the assistant header and, thinking off, the empty think block."""
    probe = [{"role": "user", "content": "x"}]
    on = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
    off = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, **TEMPLATE_KWARGS)
    return on[len(off):]


def tool_response_text(results: list[str], prefix: str) -> str:
    """Everything between the policy's <|im_end|> and its next turn."""
    body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
    return "\n<|im_start|>user" + body + "<|im_end|>\n" + prefix


class Episode:
    """One multi-turn episode. ``ids`` is everything after the prompt; ``policy`` marks the
    policy's own tokens (False = written by the harness: tool responses and turn headers)."""

    def __init__(self, case: dict, prompt_ids: list[int]):
        self.case, self.prompt_ids = case, prompt_ids
        self.ids: list[int] = []
        self.policy: list[bool] = []
        self.turns: list[str] = []
        self.turn_starts: list[int] = []  # index into ids of each turn's first token
        self.calls: list[dict] = []       # {"name", "arguments", "result", "turn"}
        self.final_text = ""
        self.truncated = False            # a turn hit the token budget before <|im_end|>
        self.unanswered = 0               # calls left when the rounds ran out
        self.answer_closed = False

    @property
    def policy_tokens(self) -> int:
        return sum(self.policy)

    def to_json(self, tokenizer=None) -> dict:
        return {"id": self.case["id"], "family": self.case["family"], "turns": self.turns,
                "calls": [{k: c[k] for k in ("name", "arguments", "turn")} | {"error": result_is_error(c["result"])}
                          for c in self.calls],
                "final_text": self.final_text, "truncated": self.truncated, "unanswered": self.unanswered,
                "answer_closed": self.answer_closed, "policy_tokens": self.policy_tokens,
                "total_tokens": len(self.prompt_ids) + len(self.ids)}


#: Largest prefill (rows x padded length) one generate call may take. A late round of 16 episodes
#: at ~7k tokens each asked for 3.8 GiB more than the 13,000 MiB cap allows; rows are split by size.
MAX_PREFILL_TOKENS = 40_000
#: 07: on Apple silicon the GPU shares the machine's memory, so a round's 8 training episodes (~6-7k tokens each)
#: go in one call instead of two (the decode loop costs about the same per step for 8 rows as for 5); kept
#: under the allocator's cap with full attention's prefill scores materialised.
MAX_PREFILL_TOKENS_MPS = 56_000


def _generate_round(policy, inputs: list[list[int]], budgets: list[int], **kw) -> list[list[int]]:
    cap = MAX_PREFILL_TOKENS_MPS if next(policy.parameters()).device.type == "mps" else MAX_PREFILL_TOKENS
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
    out: dict[int, list[int]] = {}
    chunk: list[int] = []
    for i in order + [None]:
        if i is not None and (len(chunk) + 1) * len(inputs[i]) <= cap:
            chunk.append(i)
            continue
        if chunk:
            rows = _generate_chunk(policy, [inputs[j] for j in chunk], [budgets[j] for j in chunk], **kw)
            out.update(zip(chunk, rows))
        chunk = [i] if i is not None else []
    return [out[i] for i in range(len(inputs))]


def _generate_chunk(policy, inputs: list[list[int]], budgets: list[int], *, pad: int, temperature: float,
                    im_end: int, suppress: int, stop_at_json: bool, tokenizer, cache: dict) -> list[list[int]]:
    import torch
    from transformers import StoppingCriteriaList
    device = next(policy.parameters()).device
    width = bucketed(max(len(x) for x in inputs), device)  # 07: on MPS a few prompt widths, not one per round
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in inputs], device=device)
    attention = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], device=device)
    sample = temperature > 0
    stopping = StoppingCriteriaList([_StopAtAnswer(tokenizer, width, len(inputs), cache)]) if stop_at_json else None
    out = policy.generate(input_ids=input_ids, attention_mask=attention, do_sample=sample,
                          temperature=temperature if sample else None, top_p=1.0 if sample else None,
                          top_k=0 if sample else None, max_new_tokens=max(budgets), pad_token_id=pad,
                          eos_token_id=im_end, suppress_tokens=[suppress], stopping_criteria=stopping)
    rows = [row[:b] for row, b in zip(out[:, width:].tolist(), budgets)]
    del out
    release_device_cache()
    return rows


def _tok_text(tokenizer, t: int, cache: dict) -> str:
    s = cache.get(t)
    if s is None:
        s = cache[t] = tokenizer.decode([t])
    return s


def answer_close_index(tokens: list[int], tokenizer, cache: dict) -> int | None:
    """Index of the token closing the first top-level JSON object of a turn with no <tool_call>."""
    text, ends = "", []
    for t in tokens:
        text += _tok_text(tokenizer, t, cache)
        ends.append(len(text))
    if "<tool_call>" in text:
        return None
    spans = json_spans(text)
    if not spans:
        return None
    close = spans[0][1]
    return next(k for k, e in enumerate(ends) if e >= close)


class _StopAtAnswer:
    """generate stopping criterion: a row is done once a call-free turn's first JSON object closes."""

    def __init__(self, tokenizer, width: int, rows: int, cache: dict):
        self.tokenizer, self.width, self.cache = tokenizer, width, cache
        self.texts = [""] * rows
        self.done = [False] * rows
        self.seen = width

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        new = input_ids[:, self.seen:].tolist()
        self.seen = input_ids.shape[1]
        for i, toks in enumerate(new):
            if self.done[i]:
                continue
            self.texts[i] += "".join(_tok_text(self.tokenizer, t, self.cache) for t in toks)
            if "<tool_call>" not in self.texts[i] and json_spans(self.texts[i]):
                self.done[i] = True
        return torch.tensor(self.done, device=input_ids.device)


def rollout(policy, tokenizer, cases: list[dict], pool: ToolPool, *, temperature: float, max_new_tokens: int,
            stop_at_json: bool = False, prefill: dict[str, list[dict]] | None = None) -> list[Episode]:
    """Run a batch of episodes with the simulators in the loop.

    ``max_new_tokens`` caps the policy's own tokens summed over its turns. Each round, every live
    episode writes one turn; its calls (up to MAX_CALLS per episode) run on the pool and come back as
    one user turn. A turn without calls is the answer. With ``stop_at_json`` that turn ends where its
    first JSON object closes.

    07: ``prefill`` (case id -> turns, each {"calls", "results"} and an optional leading "text") starts an
    episode from turns the harness writes -- the slipped turns of a recovery episode, with the simulator's
    errors (``probe --forced 1``). They are context, not the policy's tokens; their calls are the episode's
    calls (scored, and counted against the budgets), and their rounds count against MAX_ROUNDS.
    """
    import torch
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    suppress = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    prefix = generation_prefix(tokenizer)
    cache: dict[int, str] = {}
    eps = [Episode(c, tokenizer(render_prompt(tokenizer, c), add_special_tokens=False).input_ids) for c in cases]
    used = [0] * len(eps)  # tool rounds already taken by prefilled turns
    for i, ep in enumerate(eps):
        for turn in (prefill or {}).get(ep.case["id"], []):
            text = turn_text(turn)
            ep.turn_starts.append(len(ep.ids))
            ep.turns.append(text)
            for c, r in zip(turn["calls"], turn["results"]):
                ep.calls.append({"name": c["name"], "arguments": c["arguments"], "result": r, "turn": len(ep.turns) - 1})
            written = tokenizer(text, add_special_tokens=False).input_ids + [im_end]
            env = tokenizer(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
            ep.ids += written + env
            ep.policy += [False] * (len(written) + len(env))
            used[i] += 1
    live = list(range(len(eps)))
    with torch.no_grad():
        for rnd in range(MAX_ROUNDS + 1):
            budgets = {i: max_new_tokens - eps[i].policy_tokens for i in live}
            for i in live:
                if budgets[i] <= 0:
                    eps[i].truncated = True  # the token budget ran out between turns: no answer
            live = [i for i in live if budgets[i] > 0]
            if not live:
                break
            rows = _generate_round(policy, [eps[i].prompt_ids + eps[i].ids for i in live], [budgets[i] for i in live],
                                   pad=tokenizer.pad_token_id, temperature=temperature, im_end=im_end,
                                   suppress=suppress, stop_at_json=stop_at_json, tokenizer=tokenizer, cache=cache)
            pending = []
            for row, i in zip(rows, live):
                ep = eps[i]
                cut = next((k for k, t in enumerate(row) if t == im_end), None)
                closed = answer_close_index(row if cut is None else row[:cut], tokenizer, cache) if stop_at_json else None
                if closed is not None:
                    take, ep.answer_closed = row[: closed + 1], True
                else:
                    take = row if cut is None else row[: cut + 1]
                ep.turn_starts.append(len(ep.ids))
                ep.ids += take
                ep.policy += [True] * len(take)
                text = tokenizer.decode(take, skip_special_tokens=True)
                ep.turns.append(text)
                if closed is not None:
                    ep.final_text = text
                    continue
                if cut is None:
                    ep.truncated = True
                    ep.final_text = text
                    continue
                calls = parse_calls(text)
                if not calls:
                    ep.final_text = text
                    continue
                if rnd + used[i] >= MAX_ROUNDS:
                    ep.unanswered = len(calls)
                    continue
                pending.append((i, calls))
            # Run every pending call of the round on the pool at once.
            jobs, where = [], []
            for i, calls in pending:
                n_sims = sum(x.get("name") not in LOCAL_TOOLS for x in eps[i].calls)
                n_conv = sum(x.get("name") == CONVERT for x in eps[i].calls)
                n_comp = sum(x.get("name") == COMPARE for x in eps[i].calls)
                for c in calls:
                    if c.get("malformed"):
                        c["result"] = MALFORMED_CALL_ERROR
                    elif c.get("name") == CONVERT:
                        n_conv += 1
                        if n_conv > MAX_CONVERTS:
                            c["result"] = CONVERT_BUDGET_ERROR
                        else:
                            jobs.append((c["name"], c["arguments"]))
                            where.append(c)
                    elif c.get("name") == COMPARE:
                        n_comp += 1
                        if n_comp > MAX_COMPARES:
                            c["result"] = COMPARE_BUDGET_ERROR
                        else:
                            jobs.append((c["name"], c["arguments"]))
                            where.append(c)
                    else:
                        n_sims += 1
                        if n_sims > MAX_CALLS:
                            c["result"] = CALL_BUDGET_ERROR
                        else:
                            jobs.append((c["name"], c["arguments"]))
                            where.append(c)
            for c, result in zip(where, pool.call_many(jobs)):
                c["result"] = result
            live = []
            for i, calls in pending:
                ep = eps[i]
                for c in calls:
                    c["turn"] = len(ep.turns) - 1
                    ep.calls.append(c)
                env = tokenizer(tool_response_text([c["result"] for c in calls], prefix), add_special_tokens=False).input_ids
                ep.ids += env
                ep.policy += [False] * len(env)
                live.append(i)
    return eps


def score_episode(ep: Episode) -> dict:
    calls = [{"name": c["name"], "arguments": c["arguments"], "result": c["result"]} for c in ep.calls]
    return score(ep.case, calls, ep.final_text)


def load_policy(adapter: str | Path | None = None, device: str | None = None, merge: bool = False):
    """The model (and a LoRA adapter) for generation only. ``merge`` folds the adapter into the weights (07's probes
    on MPS: a decode step at 8 rows is 44 ms merged against 65 ms with the adapter's extra matmuls)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    patch_delta_rule(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).to(device)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def _sim_errors(calls: list[dict]) -> list[tuple[dict, bool]]:
    """An episode's simulate calls, each with whether it errored -- read from its result, or from the
    "error" flag a recorded episode (``Episode.to_json``) keeps instead."""
    out = []
    for c in calls:
        if c.get("name") in LOCAL_TOOLS:
            continue
        out.append((c, bool(c["error"]) if "result" not in c and "error" in c else result_is_error(c.get("result"))))
    return out


def recovery_stats(calls: list[dict]) -> dict:
    """07: what an episode did about its erroring simulate calls. ``erred``: a simulate call errored;
    ``recovered``: a simulate call after the first erroring one ran clean; ``repeats``: simulate calls
    identical (tool and arguments) to an earlier one; ``streak``: the longest run of erroring simulate calls."""
    sims = _sim_errors(calls)
    errs = [e for _, e in sims]
    first = errs.index(True) if True in errs else None
    seen, repeats, streak, run = set(), 0, 0, 0
    for c, e in sims:
        key = json.dumps([c.get("name"), c.get("arguments")], sort_keys=True, default=str)
        repeats += key in seen
        seen.add(key)
        run = run + 1 if e else 0
        streak = max(streak, run)
    return {"erred": first is not None, "recovered": first is not None and not all(errs[first:]),
            "repeats": repeats, "streak": streak}


def recovery_block(stats: list[dict], scores: list[dict] | None = None) -> dict:
    """The recovery measures over episodes (``scores`` adds what the erring episodes went on to earn)."""
    n = max(1, len(stats))
    erred = [i for i, r in enumerate(stats) if r["erred"]]
    k = len(erred)
    out = {"sim_erred": len(erred) / n,
           "recovered_given_error": sum(stats[i]["recovered"] for i in erred) / k if k else None,
           "repeats_per_episode": sum(r["repeats"] for r in stats) / n,
           "streak3_share": sum(r["streak"] >= 3 for r in stats) / n,
           "streak_max": max((r["streak"] for r in stats), default=0)}
    if scores is not None:
        out["evidence_given_error"] = sum(bool(scores[i]["evidence"]) for i in erred) / k if k else None
        out["decision_given_error"] = sum(bool(scores[i]["decision_correct"]) for i in erred) / k if k else None
        clean = [i for i in range(len(stats)) if not stats[i]["erred"]]
        out["decision_given_no_error"] = sum(bool(scores[i]["decision_correct"]) for i in clean) / len(clean) if clean else None
    return out


def summarise(episodes: list[Episode], scores: list[dict]) -> dict:
    """What the policy did and what it earned, overall and per family."""
    def block(pairs):
        n = len(pairs)
        eps_, scs = [p[0] for p in pairs], [p[1] for p in pairs]
        mean = lambda xs: sum(xs) / n  # noqa: E731
        all_calls = [c for e in eps_ for c in e.calls if c.get("name") not in LOCAL_TOOLS]
        converts = [c for e in eps_ for c in e.calls if c.get("name") == CONVERT]
        compares = [c for e in eps_ for c in e.calls if c.get("name") == COMPARE]
        comps = defaultdict(list)
        for s in scs:
            for k, v in s["components"].items():
                comps[k].append(v)
        return {
            "n": n, "reward": mean([s["total"] for s in scs]),
            "components": {k: sum(v) / len(v) for k, v in comps.items()},
            "called_any": mean([bool(e.calls) for e in eps_]),
            "first_call_right_tool": mean([next((c["name"] for c in e.calls if c.get("name") not in LOCAL_TOOLS), None)
                                           == e.case["tool"] for e in eps_]),
            "calls_per_episode": mean([sum(c.get("name") not in LOCAL_TOOLS for c in e.calls) for e in eps_]),
            "converts_per_episode": mean([sum(c.get("name") == CONVERT for c in e.calls) for e in eps_]),
            "converts_errored": sum(result_is_error(c["result"]) for c in converts) / max(1, len(converts)),
            "compares_per_episode": mean([sum(c.get("name") == COMPARE for c in e.calls) for e in eps_]),
            "compares_errored": sum(result_is_error(c["result"]) for c in compares) / max(1, len(compares)),
            "calls_malformed": sum(bool(c.get("malformed")) for c in all_calls) / max(1, len(all_calls)),
            "calls_errored": sum(result_is_error(c["result"]) for c in all_calls) / max(1, len(all_calls)),
            "evidence": mean([s["evidence"] for s in scs]),
            "answer_found": mean([extract_answer(e.final_text)[0] is not None for e in eps_]),
            "decision_correct": mean([s["decision_correct"] for s in scs]),
            "truncated": mean([e.truncated for e in eps_]),
            "unanswered": mean([e.unanswered > 0 for e in eps_]),
            "policy_tokens": mean([e.policy_tokens for e in eps_]),
            "total_tokens": mean([len(e.prompt_ids) + len(e.ids) for e in eps_]),
            "recovery": recovery_block([recovery_stats(e.calls) for e in eps_], scs),
        }
    pairs = list(zip(episodes, scores))
    fams = defaultdict(list)
    for p in pairs:
        fams[p[0].case["family"]].append(p)
    return {"overall": block(pairs), "families": {f: block(v) for f, v in sorted(fams.items())}}


# ---------------------------------------------------------------------------------------------
# Reference episodes for a seed: a strategy a policy could run without knowing the answer.
#
# The proving points depend on where the answer is (a scan's answer and its failing neighbour), so
# an episode that simulates only them teaches the model to guess the boundary. The reference
# strategy here is blind: simulate every candidate in one turn, then answer. It needs 1-5 calls,
# costs only efficiency (0.963-1.000 on train), and leaves pruning for PPO to find.
#
# Arguments are written the way a careful person would type them -- the tool-unit value to four
# significant figures (at most 0.05% off; the argument tolerance is 0.5%) -- and the answer is
# built from what the simulator returned for those arguments.
# ---------------------------------------------------------------------------------------------

def _typed(v: Any) -> Any:
    return float(f"{v:.4g}") if isinstance(v, float) else v


def convert_plan(case: dict, limits: bool = True) -> list[dict]:
    """Every conversion a blind episode needs, in a fixed order, each once: the shared inputs, then
    each candidate's own inputs, then the limits (``limits=False``: not the limits -- 06's check_limits
    takes them as stated and converts them itself). ``targets`` says where the converted number goes."""
    plan: dict[tuple, dict] = {}

    def add(user_value: float, u: dict, from_name: str, to_sym: str, target: tuple) -> None:
        if not needs_conversion(u):
            return
        key = (float(user_value), from_name, to_sym)  # a float: the chat template renders parsed numbers so
        plan.setdefault(key, {"value": float(user_value), "from_unit": UNIT_SYMBOL[from_name], "to_unit": to_sym,
                              "targets": []})["targets"].append(target)

    for k, inp in case["inputs"].items():
        add(inp["user_value"], unit_spec(inp["q"], inp["unit"]), inp["unit"], TOOL_UNIT[inp["q"]], ("arg", k, None))
    for p in case["points"]:
        for k, inp in p.get("inputs", {}).items():
            add(inp["user_value"], unit_spec(inp["q"], inp["unit"]), inp["unit"], TOOL_UNIT[inp["q"]], ("arg", k, p["key"]))
    for c in case["constraints"] if limits else []:
        add(c["user_value"], metric_unit_spec(c["metric"], c["user_unit"]), c["user_unit"],
            METRIC_TOOL_UNIT[c["metric"]], ("limit", c["metric"], None))
    return list(plan.values())


def blind_calls(case: dict, converted: dict[tuple, float] | None = None) -> list[dict]:
    """Every candidate simulated: each argument as the user stated it, or as convert_units returned
    it (``converted``: ("arg", key, point key or None) -> value)."""
    converted = converted or {}
    calls = []
    for p in case["points"]:
        args = {}
        for k, v in p["args"].items():
            c = converted.get(("arg", k, p["key"]), converted.get(("arg", k, None)))
            args[k] = _typed(v) if c is None else c
        calls.append({"name": case["tool"], "arguments": args})
    return calls


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def compare_calls(case: dict, results: list[str]) -> list[dict]:
    """The one check_limits call: every limit once ("op number unit": the number and the unit as the
    user stated them, the unit written as a convert_units call writes it) and, per simulated point keyed
    by its label (Result for a single point), the value of every limited field as the comparison line
    cites it (four significant figures) -- and, in an swro_option question, the objective's value, which
    the line cites too."""
    labels = [p["label"] if case["decision_kind"] != "passfail" else "Result" for p in case["points"]]
    lim = {c["metric"]: f"{c['op']} {c['user_value']:g} {UNIT_SYMBOL[c['user_unit']]}" for c in case["constraints"]}
    vals = {lab: {c["metric"]: float(_fmt(metric(json.loads(r), c["metric"]))) for c in case["constraints"]}
            for lab, r in zip(labels, results)}
    if case["family"] == "swro_option":
        for lab, r in zip(labels, results):
            vals[lab][case["objective"]] = float(_fmt(metric(json.loads(r), case["objective"])))
    return [{"name": COMPARE, "arguments": {"limits": lim, "values": vals}}]


def stated_limits(case: dict) -> dict[str, float]:
    """Every limit in its field's own unit, as check_limits converts it from the user's statement."""
    out = {}
    for c in case["constraints"]:
        v, err = limit_in_field_unit(c["metric"], c["user_value"], UNIT_SYMBOL[c["user_unit"]])
        assert err is None, (case["id"], c, err)
        out[c["metric"]] = v
    return out


def aggregation_line(case: dict, verdicts: list[bool], results: list[str]) -> str | None:
    """The line after the candidates: which passed, and which one the question's objective picks.
    A copy of the verdicts just written and of the objective the question states; none for pass/fail."""
    if case["decision_kind"] == "passfail":
        return None
    labels = [p["label"] for p in case["points"]]
    passing = [i for i, ok in enumerate(verdicts) if ok]
    if case["decision_kind"] == "number":
        word = case["objective"]
        chosen = labels[passing[0] if word == "lowest" else passing[-1]] if passing else "none"
        shown = ", ".join(labels[i] for i in passing) or "none"
    elif case["family"] == "ro_membrane":  # the offers are listed cheapest first
        word = "cheapest"
        chosen = labels[passing[0]] if passing else "none"
        shown = ", ".join(labels[i] for i in passing) or "none"
    else:  # swro_option: the lowest objective among the passing options
        word = "lowest"
        obj = [metric(json.loads(r), case["objective"]) for r in results]
        chosen = labels[min(passing, key=lambda i: obj[i])] if passing else "none"
        shown = ", ".join(f"{labels[i]} {_fmt(obj[i])}" for i in passing) or "none"
    return f"passing: {shown} -> {word}: {chosen}"


def reasoning_text(case: dict, results: list[str], limits: dict[str, float] | None = None) -> str:
    """One line per simulated point: each limited field against its limit (in the tool's units, as
    convert_units returned it when the user stated it in another unit); then the aggregation line."""
    limits = limits or {}
    lines, verdicts = [], []
    for p, r in zip(case["points"], results):
        m = json.loads(r)
        parts, ok = [], True
        for c in case["constraints"]:
            v = metric(m, c["metric"])
            passed = v >= c["threshold"] if c["op"] == ">=" else v <= c["threshold"]
            ok &= passed
            lim = limits.get(c["metric"], c["threshold"])
            parts.append(f"{c['metric']} {_fmt(v)} {'meets' if passed else 'breaks'} {c['op']} {lim:g}")
        if case["family"] == "swro_option":
            parts.append(f"{case['objective']} {_fmt(metric(m, case['objective']))}")
        label = p["label"] if case["decision_kind"] != "passfail" else "Result"
        lines.append(f"{label}: " + "; ".join(parts) + f" -> {'pass' if ok else 'fail'}")
        verdicts.append(ok)
    agg = aggregation_line(case, verdicts, results)
    return "\n".join(lines + ([agg] if agg else []))


def reference_answer(case: dict, results: list[str], reasoning: bool, limits: dict[str, float] | None = None) -> str:
    g = case["gold"]
    decision = g["decision"]
    if case["decision_kind"] == "number" and decision is not None:
        # The setting as the user named it ("2,100 ft²"): valid JSON, and what the contract asks for.
        decision = next(p["label"] for p in case["points"] if p["user_value"] == g["decision"])
    values = {}
    if g["values"]:
        at = next(i for i, p in enumerate(case["points"])
                  if case["decision_kind"] == "passfail" or p.get("user_value", p["label"]) == g["decision"])
        m = json.loads(results[at])
        values = {k: float(_fmt(metric(m, k))) for k in g["values"]}
    obj = json.dumps({"decision": decision, "violations": g["violations"], "values": values})
    return (reasoning_text(case, results, limits) + "\n\n" + obj) if reasoning else obj


def reference_episode(case: dict, pool: ToolPool) -> dict:
    """The blind episode: a convert turn (if an input needs converting), a simulate turn, a check
    turn, the answer. Verified: the converted numbers are within tolerance, no rounding flips a
    pass/fail, the comparison lines are the tool's lines, and the reward pays the episode in full."""
    plan = convert_plan(case, limits=not COMPARE_ON)  # 06: check_limits converts the limits itself
    turns = []
    converted: dict[tuple, float] = {}
    limits: dict[str, float] = {}
    if plan:
        calls = [{"name": CONVERT, "arguments": {"value": p["value"], "from_unit": p["from_unit"], "to_unit": p["to_unit"]}}
                 for p in plan]
        results = pool.call_many([(c["name"], c["arguments"]) for c in calls])
        for p, r in zip(plan, results):
            got = json.loads(r)
            assert "value" in got, (case["id"], p, r)
            for t in p["targets"]:
                if t[0] == "arg":
                    converted[t] = got["value"]
                else:
                    limits[t[1]] = got["value"]
        turns.append({"calls": calls, "results": results})
    sims = blind_calls(case, converted)
    results = pool.call_many([(c["name"], c["arguments"]) for c in sims])
    turns.append({"calls": sims, "results": results})
    # 06: one check_limits call -- every limit once as the user stated it, every simulated point's
    # limited values once; the comparison lines are copies of the lines it returns.
    labels = [p["label"] if case["decision_kind"] != "passfail" else "Result" for p in case["points"]]
    compares = compare_calls(case, results) if COMPARE_ON else []
    if compares:
        verdicts = pool.call_many([(c["name"], c["arguments"]) for c in compares])
        got = json.loads(verdicts[0])
        assert isinstance(got, list) and len(got) == len(labels), (case["id"], labels, verdicts)
        limits = stated_limits(case)
        turns.append({"calls": compares, "results": verdicts})
    by_key = {p["key"]: p for p in case["points"]}
    for c, p, r in zip(sims, case["points"], results):
        for k, v in p["args"].items():  # every argument within tolerance of the scenario's
            assert _same_arg(c["arguments"][k], v), (case["id"], p["key"], k, c["arguments"][k], v)
        seen, gold = json.loads(r), p["metrics"]
        for con in case["constraints"]:  # rounding the arguments must not flip a pass/fail
            ok = lambda x: x >= con["threshold"] if con["op"] == ">=" else x <= con["threshold"]  # noqa: E731
            assert ok(metric(seen, con["metric"])) == ok(gold[con["metric"]]), (case["id"], p["key"], con["metric"])
    for con in case["constraints"]:  # a converted limit is the threshold, to 6 significant figures
        if con["metric"] in limits:
            assert abs(limits[con["metric"]] - con["threshold"]) <= 1e-5 * abs(con["threshold"]), (case["id"], con)
    all_calls = [c | {"result": r} for t in turns for c, r in zip(t["calls"], t["results"])]
    for reasoning in (False, True):
        answer = reference_answer(case, results, reasoning, limits)
        sc = score(case, all_calls, answer)
        assert sc["decision_correct"] and sc["evidence"], (case["id"], reasoning)
    reasoned = reference_answer(case, results, True, limits)
    lines = reasoned.split("\n\n")[0]
    if compares:  # the comparison lines are the tool's lines, byte for byte
        assert lines.splitlines()[: len(labels)] == got, (case["id"], lines, got)
    agg = reasoned.split("\n\n")[0].splitlines()[-1]
    if case["decision_kind"] != "passfail":  # the aggregation line names the gold decision
        chosen = agg.rsplit(": ", 1)[1]
        gold = case["gold"]["decision"]
        want = "none" if gold is None else (next(p["label"] for p in case["points"] if p.get("user_value", p["label"]) == gold))
        assert chosen == want, (case["id"], agg, gold)
    return {"id": case["id"], "turns": turns, "answer": reference_answer(case, results, False, limits),
            "answer_reasoned": reasoned, "reward": sc["total"], "converts": len(plan), "compares": len(compares)}


def build_references(split: str, pool: ToolPool) -> list[dict]:
    """Blind reference episodes for a split, each verified against the reward."""
    return [reference_episode(case, pool) for case in load_split(split)]


# ---------------------------------------------------------------------------------------------
# 07: recovery episodes. 06's seeds slipped one input past convert_units -- the question's number passed
# as it stands -- and had never seen an erroring call fixed. A recovery episode is the blind reference
# episode with that slip made and then fixed: the slipped turns (the convert turn without that input, the
# simulate turn with the question's number in it, every call erring) are context; the fix -- one line
# naming the input, its number and the unit the question gives it in, then the convert_units call for it
# -- the simulate turn again, the check turn and the answer are the reference's own, and labelled.
#
# The line reads the question and the call, not the error: the simulator says which input is out of
# range for some slips ("feed_nacl_mass_frac must be between 0 and 1"; "fixed value 1.08 outside bounds"
# for A in LMH/bar) and only "Unit model fs.unit failed to initialize" for others (a pressure in MPa, a
# flow in kg/h, an area in ft²), and the fix is the same: find the number still in the question's unit.
# ---------------------------------------------------------------------------------------------

def _raw_values(case: dict, param: str) -> dict[str | None, float]:
    """The question's own numbers for one input, as a slip passes them: {point key, or None when shared: value}.
    Only numbers stated in a unit that needs converting (the generator states a per-candidate input in one unit
    for all the candidates of a question, so this keeps them all)."""
    if param in case["inputs"]:
        return {None: float(case["inputs"][param]["user_value"])}
    return {p["key"]: float(p["inputs"][param]["user_value"]) for p in case["points"] if param in p.get("inputs", {})
            and needs_conversion(unit_spec(p["inputs"][param]["q"], p["inputs"][param]["unit"]))}


def _stated_input(case: dict, param: str) -> dict:
    return case["inputs"].get(param) or next(p["inputs"][param] for p in case["points"] if param in p.get("inputs", {}))


def slip_params(case: dict) -> list[str]:
    """The inputs a slip can leave unconverted: each input whose stated unit needs converting and whose
    convert calls convert nothing else (a call shared with another input would slip two)."""
    owners: dict[str, set] = defaultdict(set)
    for p in convert_plan(case, limits=not COMPARE_ON):
        params = {t[1] for t in p["targets"] if t[0] == "arg"}
        for k in params:
            owners[k] |= params
    return sorted(k for k, ks in owners.items() if ks == {k})


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fix_line(case: dict, param: str, raw: list[float]) -> str:
    """The fix's own line: the tool that failed, the input, the number(s) the call gave it -- written as the
    call wrote them -- and the unit the question states it in, against the unit the tool takes."""
    inp = _stated_input(case, param)
    many = len(raw) > 1
    return (f"{case['tool']} failed: {param} {_and_list([str(v) for v in raw])} {'are' if many else 'is'} in "
            f"{UNIT_SYMBOL[inp['unit']]}, as the question gives {'them' if many else 'it'}, not {TOOL_UNIT[inp['q']]}. "
            f"Converting {'them' if many else 'it'} first.")


def error_kind(text: str) -> str:
    """What a simulator error says, coarsely: whether it names the input or only that the model failed."""
    t = text or ""
    if "must be" in t or "must exceed" in t:
        return "validation (names the input)"
    if "outside bounds" in t:
        return "out of bounds (names the variable)"
    if "failed to initialize" in t:
        return "failed to initialize"
    if "did not converge" in t:
        return "did not converge"
    return "other"


def recovery_episode(case: dict, pool: ToolPool) -> dict | None:
    """The blind reference episode with one input slipped and fixed. The input is drawn per question (its
    own seed) among ``slip_params``; one whose slip the simulator runs on any call -- a silently wrong
    number, not an error to recover from -- is passed over for the next. None when no input qualifies."""
    params = slip_params(case)
    if not params:
        return None
    random.Random(f"recover:{case['id']}").shuffle(params)
    ref = reference_episode(case, pool)
    plan = convert_plan(case, limits=not COMPARE_ON)
    conv_turn = ref["turns"][0] if plan else None
    sim_turn, rest_turns = (ref["turns"][1], ref["turns"][2:]) if plan else (ref["turns"][0], ref["turns"][1:])
    for param in params:
        mine = [j for j, p in enumerate(plan) if any(t[0] == "arg" and t[1] == param for t in p["targets"])]
        rest = [j for j in range(len(plan)) if j not in mine]
        converted = {}
        for j in rest:
            got = json.loads(conv_turn["results"][j])
            for t in plan[j]["targets"]:
                converted[t] = got["value"]
        raw = _raw_values(case, param)
        slipped = blind_calls(case, {**converted, **{("arg", param, key): v for key, v in raw.items()}})
        errors = pool.call_many([(c["name"], c["arguments"]) for c in slipped])
        if not all(result_is_error(r) for r in errors):
            continue
        turns = []
        if rest:
            turns.append({"calls": [conv_turn["calls"][j] for j in rest], "results": [conv_turn["results"][j] for j in rest],
                          "label": False})
        turns.append({"calls": slipped, "results": errors, "label": False})
        turns.append({"text": fix_line(case, param, list(raw.values())), "calls": [conv_turn["calls"][j] for j in mine],
                      "results": [conv_turn["results"][j] for j in mine], "label": True})
        turns += [dict(sim_turn, label=True)] + [dict(t, label=True) for t in rest_turns]
        all_calls = [c | {"result": r} for t in turns for c, r in zip(t["calls"], t["results"])]
        sc = score(case, all_calls, ref["answer_reasoned"])
        assert sc["decision_correct"] and sc["evidence"], (case["id"], param)
        inp = _stated_input(case, param)
        return {"id": case["id"], "turns": turns, "answer": ref["answer"], "answer_reasoned": ref["answer_reasoned"],
                "reward": sc["total"], "reference_reward": ref["reward"], "converts": len(plan), "compares": ref["compares"],
                "slip": {"param": param, "q": inp["q"], "unit": inp["unit"],
                         "level": unit_spec(inp["q"], inp["unit"])["level"], "values": list(raw.values()),
                         "calls": len(slipped), "error": errors[0], "error_kind": error_kind(errors[0]),
                         "tried": params.index(param) + 1}}
    return None


def build_recoveries(split: str, pool: ToolPool) -> tuple[list[dict], Counter]:
    """Recovery episodes for every question of a split that has a slip which errs, and why the others have none."""
    out, why = [], Counter()
    for case in load_split(split):
        if not slip_params(case):
            why["no input needs converting" if not convert_plan(case, limits=not COMPARE_ON)
                else "no input converted alone"] += 1
            continue
        ep = recovery_episode(case, pool)
        if ep is None:
            why["every slip runs (no error)"] += 1
            continue
        out.append(ep)
        why["recovery episode"] += 1
    return out, why


def _content_spans(answer: str, passfail: bool, wide: bool) -> list[tuple[int, int]]:
    """Character spans of the answer's content, as opposed to its scaffolding.

    ``wide=False`` (what the seed leaves unlabelled): the text inside a string decision, the literal
    of a null/number decision, the text of each violation, each number -- JSON syntax (quotes,
    brackets, commas) is always scaffolding. `seed-verdict` left the decision's quotes and the
    pass/fail violations list whole, and wrote `"decision": C` and violations as objects and prose.

    ``wide=True`` (what PPO may move): the whole decision literal, quotes included, and in a
    pass/fail question the whole violations list -- null versus "A", or [] versus a list, is itself
    the decision.
    """
    obj, span, _ = extract_answer(answer)
    s0 = span[0]
    text = answer[s0:span[1]]
    spans = []
    m = re.search(r'"decision"\s*:\s*', text)
    p = m.end()
    if text.startswith('"', p):
        e = p + 1
        while text[e] != '"':
            e += 2 if text[e] == "\\" else 1
        spans.append((s0 + p, s0 + e + 1) if wide else (s0 + p + 1, s0 + e))
    else:
        lit = re.match(r'[^,}\]]+', text[p:])
        if lit:
            spans.append((s0 + p, s0 + p + lit.end()))
    # A sampled answer need not have a violations list (`ppo-s0` crashed at step 34 on one without).
    vm = re.search(r'"violations"\s*:\s*\[', text)
    if vm and "]" in text[vm.end():]:
        vs, ve = vm.end(), text.index("]", vm.end())
        if wide and passfail:
            spans.append((s0 + vm.end() - 1, s0 + ve + 1))
        else:
            for im in re.finditer(r'"((?:[^"\\]|\\.)*)"', text[vs:ve]):
                spans.append((s0 + vs + im.start(1), s0 + vs + im.end(1)))
    if '"values"' not in text:
        return spans
    base = text.index('"values"')
    for km in re.finditer(r'": (-?[\d.eE+-]+)', text[base:]):
        spans.append((s0 + base + km.start(1), s0 + base + km.end(1)))
    return spans


_NUM = r"-?[\d.]+(?:e[-+]?\d+)?"


def _reasoning_content_spans(text: str, verdicts: bool) -> list[tuple[int, int]]:
    """Content of the comparison lines: every simulated value, and with ``verdicts`` also every limit,
    each verdict (meets/breaks, pass/fail) and the aggregation line's set and pick. The candidate
    label, the field names and the operators are scaffolding.

    The seed labels the verdicts (``verdicts=False`` here): `seed-scaffold` left them out, wrote "->"
    where "meets"/"breaks" goes on 719 of 720 greedy dev answers, and looped metric, value and limit
    to the token budget on up to 22% of them. PPO keeps them trainable (``verdicts=True``): they are
    the judgement it has to learn, not scaffolding to hold still."""
    spans = []
    for m in re.finditer(rf"(\S+) ({_NUM}) (meets|breaks) (<=|>=) ({_NUM})", text):
        # 05: the seed labels the limit too -- it is a copy of the prompt's number or of what
        # convert_units returned, and 04's policy copied the user's number unconverted on 91 of 102
        # limits that needed converting. 06: the value too -- the line copies it, with the verdict,
        # from the check_limits response. PPO keeps value, limit and verdict (``verdicts=True``).
        spans += [m.span(2), m.span(5), m.span(3)] if verdicts else []
    for m in re.finditer(rf"(costing\.\S+) ({_NUM})(?=\s*->)", text):
        spans.append(m.span(2))
    if verdicts:
        for m in re.finditer(r"-> (pass|fail)", text):
            spans.append(m.span(1))
        # The aggregation line: the passing set and the pick are content; "passing:" and the
        # objective word are scaffolding. The seed labels the whole line (it is a copy of the
        # verdicts above it and of the objective the question states).
        for m in re.finditer(r"^passing: (.+?) -> \w+: (.+)$", text, flags=re.M):
            spans += [m.span(1), m.span(2)]
    return spans


def decision_class(case: dict) -> str | None:
    """The answer class a label seed balances over; None for the pass/fail families (both answers
    are sampled by the scaffold seed already)."""
    if case["decision_kind"] == "passfail":
        return None
    if case["gold"]["decision"] is None:
        return "none"
    return "setting" if case["decision_kind"] == "number" else str(case["gold"]["decision"])


def pick_label_cases(k: int, seed: int) -> list[str]:
    """k train records per (family, answer class), for a label seed."""
    groups = defaultdict(list)
    for c in load_split("train"):
        cls = decision_class(c)
        if cls is not None:
            groups[(c["family"], cls)].append(c["id"])
    rng = random.Random(seed)
    return sorted(i for key in sorted(groups) for i in rng.sample(groups[key], min(k, len(groups[key]))))


def seed_example(case: dict, ref: dict, tokenizer, *, label: str, reasoning: bool, label_decision: bool = False):
    """(ids, labels) for one reference episode. Prompt and tool responses are context. The calls
    turn and its <|im_end|> are labelled. In the answer turn, ``label="full"`` labels everything;
    ``label="scaffold"`` labels only the scaffolding and leaves the decision, the violations and the
    numbers -- the task's content -- to PPO (the reasoning lines, if any, are content too).

    07: a turn marked ``"label": False`` (a recovery episode's slipped turns) is context like a tool
    response, and a turn's own line (the fix's) is written before its calls, as the template writes it."""
    import torch
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prefix = generation_prefix(tokenizer)
    answer = ref["answer_reasoned"] if reasoning else ref["answer"]
    prompt = tokenizer(render_prompt(tokenizer, case), add_special_tokens=False).input_ids
    ids, lab = list(prompt), [False] * len(prompt)
    for turn in ref["turns"]:  # each calls turn and its <|im_end|> labelled (07: unless marked not); the tool responses context
        call_ids = tokenizer(turn_text(turn), add_special_tokens=False).input_ids + [im_end]
        env = tokenizer(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
        ids += call_ids + env
        lab += [bool(turn.get("label", True))] * len(call_ids) + [False] * len(env)
    enc = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ans_ids = enc.input_ids + [im_end]
    if label == "full":
        ans_lab = [True] * len(ans_ids)
    else:
        content = _content_spans(answer, case["decision_kind"] == "passfail", wide=False)
        if label_decision:  # a label-seed record: the decision itself is labelled too
            content = content[1:]
        if reasoning:
            content += _reasoning_content_spans(answer[: extract_answer(answer)[1][0]], verdicts=False)
        ans_lab = [not any(a < e and b > s for s, e in content) for a, b in enc.offset_mapping] + [True]
    ids += ans_ids
    lab += ans_lab
    ids_t = torch.tensor([ids])
    return ids_t, torch.where(torch.tensor([lab]), ids_t, torch.full_like(ids_t, -100))


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                "out_proj")


def sft_loss(policy, ids, labels):
    """Mean next-token cross-entropy, the 248k-wide head applied only where a label is (03's)."""
    import torch
    model = policy.get_base_model()
    hidden = model.model(input_ids=ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    where = targets != -100
    return torch.nn.functional.cross_entropy(model.lm_head(hidden[where]).float(), targets[where])


def factor_fix_line(tool: str, param: str, bad: float, right: float, factor: float) -> str:
    """08: the fix line for a factor slip (the other 07's x10 / x0.1 slips): what the call gave the input against the
    value worked out from the question."""
    how = "ten times" if factor > 1 else "a tenth of"
    return f"{tool} failed: {param} {bad} is {how} the {right} worked out from the question. Calling it again with {right}."


def with_fix_line(rec: dict, case: dict) -> dict:
    """08: one of the other 07's recovery episodes with 07's fix line: the turn after the erring one opens with a line
    naming the input, the number the call gave it and what is wrong with it (07's own recovery episodes carry it; its
    seed repeated a call 0.67 times an episode on the forced probe, against the control's 4.97). The other 07's
    episodes fix without a word, and its seed repeats as often as the control."""
    turns = [dict(t) for t in rec["turns"]]
    k = next(i for i, t in enumerate(turns) if t.get("label") is False)  # the slipped turn
    bad_calls = [c for c, r in zip(turns[k]["calls"], turns[k]["results"]) if result_is_error(r)]
    slip, tool = rec["slip"], bad_calls[0]["name"]
    if slip["kind"] == "factor":
        param = slip["argument"]
        right = next(c["arguments"][param] for c in turns[k + 1]["calls"] if c["name"] == tool)
        line = factor_fix_line(tool, param, bad_calls[0]["arguments"][param], right, slip["factor"])
    else:
        param = next(p for p, v in bad_calls[0]["arguments"].items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)
                     and abs(v - slip["value"]) <= 1e-9 * max(1.0, abs(slip["value"])))
        line = fix_line(case, param, list(dict.fromkeys(c["arguments"][param] for c in bad_calls)))  # each number once
    turns[k + 1]["text"] = line
    return {**rec, "turns": turns, "fix_line": line}


def seed_refs_every(limit: int, every: int, seed: int, fix_lines: bool = False) -> list[dict]:
    """08: the seed recipe of the other notebook 07 (effa2e4: ``seed --recovery-every``): the first ``limit`` of 06's
    reference episodes in 06's shuffle, every ``every``-th replaced by its own question's recovery episode (RT_RECOVERY;
    at 300 and 4: 225 references and 75 recoveries, its dose that was alive and as good as 06's 300 seed on dev).
    ``fix_lines``: each recovery episode given 07's fix line (``with_fix_line``) -- both recipes in one seed."""
    refs = [json.loads(line) for line in reference_path("train").read_text().splitlines()]
    random.Random(seed).shuffle(refs)
    refs = refs[:limit] if limit else refs
    rec = {r["id"]: r for r in map(json.loads, RT_RECOVERY.read_text().splitlines())}
    cases = {c["id"]: c for c in load_split("train")} if fix_lines else {}
    for i in range(every - 1, len(refs), every):
        if refs[i]["id"] in rec:
            r = rec[refs[i]["id"]]
            refs[i] = with_fix_line(r, cases[r["id"]]) if fix_lines else r
    return refs


def seed_refs(limit: int, recover: int, seed: int) -> list[dict]:
    """The seed's episodes: the first ``limit`` of 06's reference episodes in one shuffle (06's selection: the
    same N picks the same episodes) and ``recover`` recovery episodes of other questions, drawn by the same
    seed, the two interleaved by a third shuffle."""
    refs = [json.loads(line) for line in reference_path("train").read_text().splitlines()]
    random.Random(seed).shuffle(refs)
    refs = refs[:limit] if limit else refs
    if recover:
        taken = {r["id"] for r in refs}
        recs = [r for r in map(json.loads, recovery_path("train").read_text().splitlines()) if r["id"] not in taken]
        random.Random(f"{seed}:recover").shuffle(recs)
        if len(recs) < recover:
            raise SystemExit(f"only {len(recs)} recovery episodes outside the first {limit}; asked for {recover}")
        refs = refs + recs[:recover]
        random.Random(f"{seed}:mix").shuffle(refs)
    return refs


def train_seed(out_dir: Path, *, label: str, reasoning: bool, epochs: int, lr: float, accum: int, lora_r: int,
               seed: int, limit: int, label_per_class: int = 0, recover: int = 0, recovery_every: int = 0,
               fix_lines: bool = False, progress=print) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device()
    patch_delta_rule(device)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    policy = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=list(LORA_TARGETS),
                                             task_type="CAUSAL_LM")).to(device)
    cases = {c["id"]: c for c in load_split("train")}
    refs = (seed_refs_every(limit, recovery_every, seed, fix_lines) if recovery_every
            else seed_refs(limit, recover, seed))
    slips = [r["slip"] for r in refs if "slip" in r]
    label_ids = set(pick_label_cases(label_per_class, seed)) if label_per_class else set()
    out_dir.mkdir(parents=True, exist_ok=True)
    recovery = ({"recovery_every": recovery_every, "recovery_source": f"effa2e4 ({RT_RECOVERY.name})",
                 "recovery_slips": dict(Counter(s.get("kind") for s in slips).most_common()),
                 "fix_lines": fix_lines} if recovery_every else
                {"recovery_slipped_units": dict(Counter(f"{s['param']} [{s['unit']}]" for s in slips).most_common()),
                 "recovery_error_kinds": dict(Counter(s["error_kind"] for s in slips).most_common())})
    (out_dir / "config.json").write_text(json.dumps(
        {"model": MODEL, "label": label, "reasoning": reasoning, "epochs": epochs, "lr": lr, "accum": accum,
         "lora_r": lora_r, "seed": seed, "examples": len(refs), "lora_targets": list(LORA_TARGETS),
         "strategy": "blind: convert what needs converting, simulate every candidate, then the answer",
         "compare_tool": COMPARE_ON, "device": device,
         "reference_episodes": len(refs) - len(slips), "recovery_episodes": len(slips),
         "recovery_ids": sorted(r["id"] for r in refs if "slip" in r), **recovery,
         "label_per_class": label_per_class, "label_case_ids": sorted(label_ids),
         "label_class_counts": dict(Counter(f"{cases[i]['family']}:{decision_class(cases[i])}" for i in label_ids))},
        indent=2) + "\n")
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    t0 = time.time()
    with (out_dir / "losses.jsonl").open("w") as fh:
        for epoch in range(epochs):
            losses = []
            for i, ref in enumerate(refs):
                ids, labels = seed_example(cases[ref["id"]], ref, tok, label=label, reasoning=reasoning,
                                           label_decision=ref["id"] in label_ids)
                extra = bucketed(ids.shape[1], device) - ids.shape[1]  # 07: on MPS, unlabelled right padding
                if extra:
                    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                    ids = torch.cat([ids, torch.full((1, extra), pad, dtype=ids.dtype)], dim=1)
                    labels = torch.cat([labels, torch.full((1, extra), -100, dtype=labels.dtype)], dim=1)
                ids, labels = ids.to(device), labels.to(device)
                loss = sft_loss(policy, ids, labels)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(refs):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    release_device_cache()
                losses.append(float(loss.detach()))
                fh.write(json.dumps({"epoch": epoch, "i": i, "loss": losses[-1], "recovery": "slip" in ref,
                                     "seconds": round(time.time() - t0, 1)}) + "\n")
                fh.flush()
            policy.save_pretrained(out_dir / f"epoch{epoch + 1}")
            k = max(1, len(losses) // 10)
            progress(f"  epoch {epoch + 1}: loss first-10% {sum(losses[:k]) / k:.4f} -> last-10% {sum(losses[-k:]) / k:.4f}"
                     f"  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------
# PPO. 03's actor-critic, carried over to multi-turn episodes with the simulators in the loop.
#
# What changes from 03: an episode has any number of call turns before its answer, the tool's
# responses are ~900 tokens each (context: no log-prob, reward or value), and the credits come from
# score()'s anchors -- each call's end, the decision, the violations, each reported number, the
# answer's closing brace. Log-probabilities are computed only at the policy's own positions: an
# episode is up to ~7k tokens and the vocabulary 248k, so full-span logits would be ~5 GB each.
# ---------------------------------------------------------------------------------------------

SUPPRESSED = -1e9


def _bounds(tokenizer, ids: list[int], cache: dict) -> list[int]:
    """End character offset (in the turn's decoded text) of each token."""
    out, n = [], 0
    for t in ids:
        n += len(_tok_text(tokenizer, t, cache)) if t not in tokenizer.all_special_ids else 0
        out.append(n)
    return out


def _at(bounds: list[int], offset: int) -> int:
    return next((i for i, e in enumerate(bounds) if e >= offset), len(bounds) - 1)


def answer_anchor_offsets(text: str, case: dict) -> dict:
    """Character offsets, in the answer turn, where each answer credit lands."""
    obj, span, _ = extract_answer(text)
    if span is None:
        return {}
    s0, region = span[0], text[span[0]: span[1]]
    out = {"close": span[1]}
    m = re.search(r'"decision"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\]]+)', region)
    if m:
        out["decision"] = s0 + m.end(1)
    m = re.search(r'"violations"\s*:\s*\[[^\]]*\]', region)
    if m:
        out["violations"] = s0 + m.end()
    for path in case["gold"]["values"]:
        m = re.search(rf'"{re.escape(path)}"\s*:\s*"?({_NUM})', region)
        if m is None and "." in path:
            sec, leaf = path.split(".", 1)
            ms = re.search(rf'"{re.escape(sec)}"\s*:\s*\{{', region)
            if ms:
                m2 = re.search(rf'"{re.escape(leaf)}"\s*:\s*"?({_NUM})', region[ms.end():])
                if m2:
                    out[("value", path)] = s0 + ms.end() + m2.end(1)
                    continue
        if m:
            out[("value", path)] = s0 + m.end(1)
    return out


def answer_content_spans(text: str, case: dict) -> list[tuple[int, int]]:
    """Content of an answer turn (what the scaffold seed left to PPO); if the JSON does not parse,
    every number and verdict word in the turn."""
    obj, span, _ = extract_answer(text)
    spans = []
    if obj is not None:
        try:
            spans += _content_spans(text[: span[1]], case["decision_kind"] == "passfail", wide=True)
        except Exception:  # noqa: BLE001 -- a malformed sampled answer falls back to the regex below
            obj = None
        spans += _reasoning_content_spans(text[: span[0]], verdicts=True)
    if obj is None:
        spans += [m.span() for m in re.finditer(rf"{_NUM}|\b(?:meets|breaks|pass|fail|null)\b", text)]
    return spans


def build_batch(episodes: list[Episode], tokenizer, device) -> dict:
    """Left-padded prompts, the episodes, right padding; the policy's tokens compacted; per-token
    rewards from score()'s credits; the masks the actor's loss uses; each policy token's turn."""
    import torch
    pad = tokenizer.pad_token_id
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    cache: dict[int, str] = {}
    width = bucketed(max(len(ep.prompt_ids) for ep in episodes), device)  # 07: rounded up on MPS (SHAPE_BUCKET_MPS)
    span = bucketed(max(1, max(len(ep.ids) for ep in episodes)), device)
    n = max(1, max(ep.policy_tokens for ep in episodes))
    rows = {k: [] for k in ("seqs", "attn", "index", "ids", "mask", "answer", "turn_end", "content", "rewards", "turn")}
    scores = []
    for ep in episodes:
        left, right = width - len(ep.prompt_ids), span - len(ep.ids)
        rows["seqs"].append([pad] * left + ep.prompt_ids + ep.ids + [pad] * right)
        rows["attn"].append([0] * left + [1] * (len(ep.prompt_ids) + len(ep.ids)) + [0] * right)
        where = [k for k, mine in enumerate(ep.policy) if mine]
        fill = n - len(where)
        mine = [ep.ids[k] for k in where]
        rows["index"].append(where + [0] * fill)
        rows["ids"].append(mine + [pad] * fill)
        rows["mask"].append([1.0] * len(where) + [0.0] * fill)
        # Each turn's policy tokens, in compacted coordinates.
        starts = [sum(ep.policy[:s]) for s in ep.turn_starts] + [len(where)]
        turns = [(starts[t], starts[t + 1]) for t in range(len(ep.turns))]
        answer, turn_end, content, reward = ([0.0] * n for _ in range(4))
        turn_of = [0] * n
        for j, (a, b) in enumerate(turns):
            for k in range(a, b):
                turn_of[k] = j
            if b > a and mine[b - 1] == im_end:
                turn_end[b - 1] = 1.0
        sc = score_episode(ep)
        scores.append(sc)
        answered = bool(ep.turns) and ep.final_text == ep.turns[-1] and not parse_calls(ep.final_text)
        if answered:
            a, b = turns[-1]
            bounds = _bounds(tokenizer, mine[a:b], cache)
            for k in range(a, b):
                answer[k] = 1.0
            starts_c = [0] + bounds[:-1]
            for s, e in answer_content_spans(ep.final_text, ep.case):
                for k in range(a, b):
                    if starts_c[k - a] < e and bounds[k - a] > s:
                        content[k] = 1.0
            offsets = answer_anchor_offsets(ep.final_text, ep.case)
        # Credits onto tokens.
        call_rank = Counter()
        for (anchor, amount) in sc["credits"]:
            if anchor[0] == "call":
                c = ep.calls[anchor[1]]
                t = c["turn"]
                k = call_rank[t]
                call_rank[t] += 1
                a, b = turns[t]
                ends = [m.end() for m in re.finditer(r"</tool_call>", ep.turns[t])]
                pos = b - 1 if k >= len(ends) else a + _at(_bounds(tokenizer, mine[a:b], cache), ends[k])
            elif answered:
                a, b = turns[-1]
                key = anchor[0] if anchor[0] != "value" else ("value", anchor[1])
                off = offsets.get(key, offsets.get("close"))
                pos = b - 1 if off is None else a + _at(_bounds(tokenizer, mine[a:b], cache), off)
            else:
                pos = len(where) - 1
            reward[max(0, pos)] += amount
        rows["answer"].append(answer)
        rows["turn_end"].append(turn_end)
        rows["content"].append(content)
        rows["rewards"].append(reward)
        rows["turn"].append(turn_of)
    t = lambda x: torch.tensor(x, device=device)  # noqa: E731
    out = {k: t(v) for k, v in rows.items()}
    out.update(span=span, scores=scores)
    return out


def padded_position_ids(attention_mask):
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def _take(x, index):
    if x.dim() == 3:
        import torch
        return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    import torch
    return torch.gather(x, 1, index)


def forward_policy(model, seqs, attn, span: int, index, *, suppress: int, hidden_layer: int | None = None):
    """Log-probs at the policy's tokens (``suppress`` removed from the vocabulary), the critic's
    features at the same positions, and the probability the model puts on ``suppress`` there."""
    import torch
    base = model.get_base_model()
    out = base.model(input_ids=seqs, attention_mask=attn, position_ids=padded_position_ids(attn),
                     output_hidden_states=hidden_layer is not None, use_cache=False)
    offset = seqs.shape[1] - span
    pos = offset + index - 1  # the position whose output predicts each policy token
    logits = base.lm_head(_take(out.last_hidden_state, pos)).float()
    mass = None
    if hidden_layer is not None:
        # Its own small tensor: indexing a softmax keeps the whole (n x 248k) result alive as a view,
        # ~0.6 GB per episode, and `gate40-s0`'s first attempt ran out of memory at step 14 on it.
        mass = torch.exp(logits[..., suppress] - torch.logsumexp(logits, dim=-1)).detach()
    logits[..., suppress] = SUPPRESSED
    targets = _take(seqs[:, offset:], index)
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    hidden = _take(out.hidden_states[hidden_layer], pos).detach() if hidden_layer is not None else None
    return logprobs, hidden, mass


def gae(rewards, values, mask, *, gamma: float = 1.0, lam: float = 1.0):
    import torch
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)
    for t in range(tokens - 1, -1, -1):
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t]
        advantages[:, t] = running
    return advantages * mask, (advantages + values) * mask


def gae_turn(rewards, values, mask, turn, *, gamma: float = 1.0, lam: float = 1.0):
    """GAE with the turn as the unit (``adv_unit`` "turn"). Per episode: a turn's reward is the sum of the
    rewards on its tokens (the credits that land in it, less its KL penalties), its value the critic's estimate
    at its first token -- the hidden state before the turn -- and the recursion runs over the turns; every token
    of a turn is given its turn's advantage and return."""
    import torch
    advantages, returns = torch.zeros_like(values), torch.zeros_like(values)
    for i in range(mask.shape[0]):
        on = mask[i] > 0
        if not bool(on.any()):
            continue
        ids = torch.unique(turn[i][on], return_inverse=True)[1]   # the turns with tokens, renumbered 0..T-1 in order
        T = int(ids.max()) + 1
        R = torch.zeros(T, dtype=values.dtype, device=values.device).index_add_(0, ids, rewards[i][on])
        active = torch.nonzero(on).squeeze(-1)
        first = torch.stack([active[ids == j][0] for j in range(T)])
        V = values[i][first]
        A = torch.zeros(T, dtype=values.dtype, device=values.device)
        running = torch.zeros((), dtype=values.dtype, device=values.device)
        for j in range(T - 1, -1, -1):
            next_value = V[j + 1] if j + 1 < T else torch.zeros((), dtype=values.dtype, device=values.device)
            running = R[j] + gamma * next_value - V[j] + gamma * lam * running
            A[j] = running
        advantages[i][on] = A[ids]
        returns[i][on] = (A + V)[ids]
    return advantages * mask, returns * mask


def position_clock(returns, mask, bins: int = 20):
    import torch
    sel = mask > 0
    out = torch.zeros_like(returns)
    index = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    binned = (index / lengths * bins).floor().clamp(max=bins - 1).long()
    for b in range(bins):
        hit = (binned == b) & sel
        if hit.any():
            out[hit] = returns[hit].mean()
    return out


def critic_fit(values, returns, mask) -> dict:
    sel = mask > 0
    v, g = values[sel], returns[sel]
    g_var = float(g.var(unbiased=False)) if g.numel() > 1 else 0.0
    if g_var < 1e-6:
        return {"value_ev": None, "value_ev_position": None}
    clock = position_clock(returns, mask)[sel]
    return {"value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6),
            "value_ev_position": round(1.0 - float((g - clock).var(unbiased=False)) / g_var, 6)}


def ppo_actor_loss(logprobs, old_logprobs, advantages, mask, *, clip_eps: float):
    import torch
    active = mask.sum().clamp(min=1.0)
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -(torch.min(unclipped, clipped) * mask).sum() / active
    with torch.no_grad():
        # 08: approx_kl, mean((r - 1) - log r) -- KL from the rollout policy to the current one, the low-variance
        # estimator the 37 details' implementation logs (and its early stop reads)
        logratio = logprobs - old_logprobs
        stats = {"actor_loss": float(loss), "ratio_mean": float((ratio * mask).sum() / active),
                 "clip_frac": float(((unclipped > clipped) & (mask > 0)).float().sum() / active),
                 "approx_kl": float((((ratio - 1) - logratio) * mask).sum() / active)}
    return loss, stats


def normalized_advantages(adv, mask):
    """08: advantages minus their mean, over their standard deviation, over the tokens ``mask`` marks (the 37 details'
    #7); zero elsewhere."""
    sel = mask > 0
    if int(sel.sum()) < 2:
        return adv
    a = adv[sel]
    return (adv - a.mean()) / (a.std() + 1e-8) * mask


def value_loss(values, old_values, returns, *, clip_eps: float | None):
    import torch
    plain = (values - returns) ** 2
    moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps) if clip_eps else values
    loss = torch.max(plain, (moved - returns) ** 2).mean()
    return loss, {"value_loss": float(loss.detach()), "value_mean": float(values.mean().detach()),
                  "return_mean": float(returns.mean())}


def _value_head(hidden_size: int, init_bias: float):
    import torch.nn as nn
    head = nn.Linear(hidden_size, 1, dtype=__import__("torch").float32)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, init_bias)
    return head


def _values(head, hidden):
    return head(hidden.float()).squeeze(-1)


def load_trainable(adapter: Path, device: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    patch_delta_rule(device)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    policy = PeftModel.from_pretrained(base, str(adapter), is_trainable=True).to(device)
    return policy, tok


# ---------------------------------------------------------------------------------------------
# 08: the critic as another model, and progress under a prover (Setlur et al. 2024, arXiv 2410.08146).
#
# The paper's verifiers (PAVs) are a separate model -- every one of them fine-tuned from the Gemma 9B pretrained
# checkpoint, for 2B, 9B and 27B policies alike -- trained to predict the prover's value from a state. Here the critic
# is a separate pretrained model too, frozen (fine-tuning a 2B model inside each PPO step is beyond this machine), read
# at the state before each turn; two linear heads on its features are refitted from scratch on the critic's window at
# every step. The prover is BoK(pi), the paper's class for its RL: its value is computed from the base policy's success
# probability, Q^{BoK(pi)} = 1 - (1 - Q^pi)^K, so the heads need only the base policy's own outcomes -- no rollouts of
# another policy (the paper's MC labels came from 20 rollouts per prefix; a rollout here is minutes of simulation).
# ---------------------------------------------------------------------------------------------

#: The candidate critics: pretrained models other than the policy. Qwen3-1.7B is another family (Qwen3: attention in
#: every layer, a 151k vocabulary; the policy is Qwen3.5's hybrid of gated delta-rule and attention layers, 248k) at 2x
#: the policy; Qwen3.5-2B is the policy's own family one size up (the paper's best prover across models was one size up:
#: 2B -> 9B, 9B -> 27B); Qwen3-4B is the paper's verifier-to-policy ratio (Gemma 9B for its 2B policy: 4.5x).
CRITICS = {"qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3.5-2b": "Qwen/Qwen3.5-2B", "qwen3-4b": "Qwen/Qwen3-4B"}
#: The decoder layers ``critic-fit`` reads from each (1-based outputs: the middle, three quarters, the last). "policy"
#: is 07's critic, for comparison: the policy's own residual stream (07's seed adapter), 07's layer 23 among them.
CRITIC_FIT_LAYERS = {"policy": (12, 18, 23), "qwen3-1.7b": (14, 21, 28), "qwen3.5-2b": (12, 18, 24),
                     "qwen3-4b": (18, 27, 36)}
CRITIC_FIT = HERE / f"{PREFIX}_critic_fit.json"


def success(sc: dict) -> bool:
    """08: the outcome whose probability the prover's value is built from: the decision right, with the simulations
    that prove it run (the reward pays the decision in full only then)."""
    return bool(sc["decision_correct"]) and bool(sc["evidence"])


def episode_text(ep: Episode, tok, prefix: str) -> tuple[str, list[int]]:
    """08: an episode as text -- the prompt as the policy saw it, each turn, the harness's tool responses -- and where
    each turn starts in it. A separate critic has its own tokenizer, so it reads the text rather than the policy's
    tokens; the state a turn starts from is the text before it."""
    text = render_prompt(tok, ep.case)
    cuts = []
    for j, turn in enumerate(ep.turns):
        cuts.append(len(text))
        text += turn
        if j + 1 < len(ep.turns):  # a turn followed by another called tools, and the harness answered them
            text += "<|im_end|>" + tool_response_text([c["result"] for c in ep.calls if c["turn"] == j], prefix)
    return text, cuts


class SeparateCritic:
    """08: a frozen model whose residual stream after decoder layer(s) ``layers`` (1-based), at the last token before
    each turn, is the critic's features. ``name`` is a key of CRITICS, or "policy" (07's critic's features: the policy
    with ``adapter`` merged). ``truncate`` drops the layers after the deepest one read (PPO reads one layer)."""

    def __init__(self, name: str, layers, device: str, adapter: Path | None = None, truncate: bool = False):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.layers, self.device = name, tuple(int(x) for x in layers), device
        patch_delta_rule(device)  # Qwen3.5's linear-attention layers on MPS (the policy, Qwen3.5-2B)
        if name == "policy":
            model, self.tok = load_policy(adapter, device=device, merge=True)
        else:
            self.tok = AutoTokenizer.from_pretrained(CRITICS[name])
            model = AutoModelForCausalLM.from_pretrained(CRITICS[name], dtype=torch.bfloat16).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.body = model.model  # the decoder stack: the LM head's logits are never computed
        self.depth = len(self.body.layers)
        if truncate:
            self.body.layers = nn.ModuleList(list(self.body.layers)[: max(self.layers)])
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.hidden_size = model.config.get_text_config().hidden_size
        self.seconds, self.tokens = 0.0, 0
        # The system turn (system prompt + tool declarations, ~2.5k of an episode's ~5k tokens) is the same in every
        # episode: Qwen3 models read it once and copy its attention cache for each episode -- the same features to
        # the bit, in about half the time. Qwen3.5's hybrid layers are read whole: continued from a cached prefix,
        # transformers' gated delta rule (chunked from another offset, in bf16) moved the features by ~3%.
        self.prefix_cache = not type(model).__name__.startswith("Qwen3_5")
        self._prefix = None

    @staticmethod
    def shared_prefix(text: str) -> str:
        """The system turn, through its <|im_end|> and newline: the text before the first user turn."""
        k = text.find("<|im_end|>\n")
        return text[: k + len("<|im_end|>\n")] if k >= 0 else ""

    def _prefix_state(self, prefix: str):
        import torch
        if self._prefix is None or self._prefix[0] != prefix:
            ids = self.tok(prefix, add_special_tokens=False).input_ids
            with torch.no_grad():
                out = self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
            self._prefix = (prefix, len(ids), out.past_key_values)
        return self._prefix

    def features(self, text: str, cuts: list[int]) -> dict[int, Any]:
        """{layer: (turns x hidden) float32 tensor on the CPU}: the state before each turn."""
        import bisect
        import copy
        import torch
        prefix = self.shared_prefix(text) if self.prefix_cache else ""
        if not prefix or cuts[0] < len(prefix):
            prefix = ""
        enc = self.tok(text[len(prefix): cuts[-1]], add_special_tokens=False, return_offsets_mapping=True)
        ends = [len(prefix) + e for _, e in enc["offset_mapping"]]
        pos = [max(0, bisect.bisect_right(ends, c) - 1) for c in cuts]
        n = pos[-1] + 1
        # Right padding to a few lengths on MPS (SHAPE_BUCKET_MPS: a new length per episode grows the graph cache for
        # good); causal attention and the causal recurrences leave every position before the padding untouched.
        ids = enc["input_ids"][:n]
        ids = ids + [self.pad] * (bucketed(n, self.device) - n)
        got: dict[int, Any] = {}
        where = torch.tensor(pos, device=self.device)
        t0 = time.perf_counter()
        cache = self._prefix_state(prefix)[2] if prefix else None  # read before the hooks go on: they index this episode

        def hook(layer: int):
            def grab(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                got[layer] = h[0].index_select(0, where).float().cpu()
            return grab

        handles = [self.body.layers[L - 1].register_forward_hook(hook(L)) for L in self.layers]
        try:
            with torch.no_grad():
                if cache is not None:
                    self.body(input_ids=torch.tensor([ids], device=self.device), past_key_values=copy.deepcopy(cache),
                              use_cache=True)
                else:
                    self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        self.seconds += time.perf_counter() - t0
        self.tokens += n
        return got


def fit_ridge(Z, y, lam: float):
    """Ridge regression with an unpenalised intercept: minimises mean (y - Zw - b)^2 + lam |w|^2 (float64)."""
    import torch
    n, d = Z.shape
    zm, ym = Z.mean(0), y.mean()
    Zc, yc = Z - zm, y - ym
    if n < d:  # the dual: w = Zc^T (Zc Zc^T + n lam I)^-1 yc
        w = Zc.T @ torch.linalg.solve(Zc @ Zc.T + n * lam * torch.eye(n, dtype=Z.dtype), yc)
    else:
        w = torch.linalg.solve(Zc.T @ Zc + n * lam * torch.eye(d, dtype=Z.dtype), Zc.T @ yc)
    return w, ym - zm @ w


def fit_logistic(Z, y, lam: float, iters: int = 50):
    """L2 logistic regression by Newton's method: minimises mean log-loss + lam/2 |w|^2, the intercept (all but)
    unpenalised (float64)."""
    import torch
    n, d = Z.shape
    X = torch.cat([Z, torch.ones(n, 1, dtype=Z.dtype)], 1)
    pen = torch.full((d + 1,), n * lam, dtype=Z.dtype)
    pen[-1] = n * 1e-6  # keeps the intercept finite when every label agrees
    w = torch.zeros(d + 1, dtype=Z.dtype)
    p0 = min(max(float(y.mean()), 1e-3), 1 - 1e-3)
    w[-1] = math.log(p0 / (1 - p0))
    for _ in range(iters):
        p = torch.sigmoid(X @ w)
        g = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p)).unsqueeze(1)).T @ X + torch.diag(pen)
        step = torch.linalg.solve(H, g)
        w = w - step
        if float(step.abs().max()) < 1e-7:
            break
    return w[:-1], w[-1]


class CriticHeads:
    """08: the separate critic's two linear heads, refitted from scratch at every step on the critic's window:
    v, ridge regression on the turn's return to go (PPO's baseline: 07's head's job), and q, L2 logistic regression on
    the episode's success (the base policy's Q^pi, from which the prover's value is computed). Features are
    standardised by the window's own statistics. Before any data: v = ``init_v``, q = ``init_q``."""

    def __init__(self, ridge: float, l2: float, init_v: float, init_q: float = 0.5):
        self.ridge, self.l2, self.init_v, self.init_q = ridge, l2, init_v, init_q
        self.state: dict | None = None

    def fit(self, X, G, y) -> None:
        if len(X) < 2:
            return
        Xd = X.double()
        mu, sd = Xd.mean(0), Xd.std(0).clamp(min=1e-6)
        Z = (Xd - mu) / sd
        wv, bv = fit_ridge(Z, G.double(), self.ridge)
        wq, bq = fit_logistic(Z, y.double(), self.l2)
        self.state = {"mu": mu, "sd": sd, "wv": wv, "bv": bv, "wq": wq, "bq": bq}

    def _z(self, X):
        return (X.double() - self.state["mu"]) / self.state["sd"]

    def value(self, X):
        import torch
        if self.state is None:
            return torch.full((len(X),), float(self.init_v))
        return (self._z(X) @ self.state["wv"] + self.state["bv"]).float()

    def success(self, X):
        import torch
        if self.state is None:
            return torch.full((len(X),), float(self.init_q))
        return torch.sigmoid(self._z(X) @ self.state["wq"] + self.state["bq"]).float()


def bok_value(q, k: int):
    """The paper's Eq. 30: the success probability of Best-of-K of a policy that succeeds with probability q."""
    return 1.0 - (1.0 - q) ** k


def prover_progress(q, y: float, k: int):
    """08: A^mu of each turn under the prover BoK(pi): f_K(q after the turn) - f_K(q before it), q the base policy's
    success probability before each turn (``q``: one per turn) and the last turn's \"after\" the episode's own outcome
    ``y`` (the state after it is terminal). Summed over the turns it telescopes to f_K(y) - f_K(q_0); the paper adds
    each step's own A^mu to that step's advantage (not its sum to go), which is how ``train_ppo`` uses it."""
    import torch
    after = torch.cat([q[1:], torch.tensor([float(y)], dtype=q.dtype)])
    return bok_value(after, k) - bok_value(q, k)


def per_turn_tokens(per_turn: list, turn, mask):
    """08: one value per turn (a tensor per episode) spread over that turn's policy tokens (``build_batch``'s rows)."""
    import torch
    out = torch.zeros_like(mask)
    for i, v in enumerate(per_turn):
        v = v.to(device=mask.device, dtype=mask.dtype)
        out[i] = v[turn[i].clamp(max=len(v) - 1)] * mask[i]
    return out


def separate_critic_stats(v_turn, q_turn, G_turn, ys, prog_turn, alpha: float) -> dict:
    """08: a step's turns as the separate critic saw them before its refit (out of sample): v against the return to
    go, q against the episode's success, and the prover's progress per turn."""
    import torch
    v, G = torch.cat(v_turn).double(), torch.cat(G_turn).double()
    q = torch.cat(q_turn).double().clamp(1e-6, 1 - 1e-6)
    y = torch.cat([torch.full((len(t),), y_, dtype=torch.float64) for t, y_ in zip(q_turn, ys)])
    p = torch.cat(prog_turn).double()
    g_var = float(G.var(unbiased=False))
    return {"value_mean": float(v.mean()), "return_mean": float(G.mean()),
            "critic_ev_turn": None if g_var < 1e-6 else 1.0 - float((G - v).var(unbiased=False)) / g_var,
            "success_rate": sum(ys) / len(ys), "q_mean": float(q.mean()),
            "q0_mean": float(torch.stack([t[0] for t in q_turn]).mean()),
            "q_brier": float(((q - y) ** 2).mean()),
            "q_logloss": float(-(y * q.log() + (1 - y) * (1 - q).log()).mean()),
            "prog_mean": float(p.mean()), "prog_std": float(p.std(unbiased=False)), "prog_abs": float(p.abs().mean()),
            "prog_alpha_std": alpha * float(p.std(unbiased=False))}


def read_jsonl(path: Path) -> list[dict]:
    """JSON lines; a partly written last line (a run still appending to the file) is skipped."""
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def replay(rows: list[dict], cases: dict[str, dict], pool: ToolPool) -> list[Episode]:
    """08: recorded episodes (``Episode.to_json`` rows: PPO's samples.jsonl, a probe's episodes) rebuilt with their
    tool results -- each call run again on the pool (the simulators are deterministic and the pool caches repeats), the
    budget and malformed-call errors given as the rollout gave them -- so that an episode's text and score are the
    rollout's. ``prompt_ids``/``ids`` stay empty: a replayed episode is read as text (``episode_text``)."""
    eps, jobs = [], []
    for r in rows:
        ep = Episode(cases[r["id"]], [])
        ep.turns, ep.final_text = list(r["turns"]), r["final_text"]
        ep.truncated, ep.unanswered, ep.answer_closed = r["truncated"], r["unanswered"], r["answer_closed"]
        earlier: list[dict] = []
        for j in range(len(ep.turns)):
            mine = [c for c in r["calls"] if c["turn"] == j]
            # The rollout's budgets: counted at the round's start over every earlier call (a malformed one counts
            # as a simulation there), then call by call within the round.
            n_sims = sum(c["name"] not in LOCAL_TOOLS for c in earlier)
            n_conv = sum(c["name"] == CONVERT for c in earlier)
            n_comp = sum(c["name"] == COMPARE for c in earlier)
            for c in mine:
                call = {"name": c["name"], "arguments": c["arguments"], "turn": j, "result": None,
                        "recorded_error": c.get("error")}
                if c["name"] is None:
                    call["result"] = MALFORMED_CALL_ERROR
                elif c["name"] == CONVERT:
                    n_conv += 1
                    call["result"] = CONVERT_BUDGET_ERROR if n_conv > MAX_CONVERTS else None
                elif c["name"] == COMPARE:
                    n_comp += 1
                    call["result"] = COMPARE_BUDGET_ERROR if n_comp > MAX_COMPARES else None
                else:
                    n_sims += 1
                    call["result"] = CALL_BUDGET_ERROR if n_sims > MAX_CALLS else None
                if call["result"] is None:
                    jobs.append(call)
                ep.calls.append(call)
            earlier += mine
        eps.append(ep)
    for b in range(0, len(jobs), 256):
        part = jobs[b: b + 256]
        for call, result in zip(part, pool.call_many([(c["name"], c["arguments"]) for c in part])):
            call["result"] = result
    return eps


def turn_rewards(ep: Episode, sc: dict) -> list[float]:
    """08: score()'s credits summed by the turn they land in (``build_batch``'s placement): a call's at its turn, the
    answer's -- or, unanswered, everything else -- at the last turn."""
    R = [0.0] * len(ep.turns)
    for anchor, amount in sc["credits"]:
        R[ep.calls[anchor[1]]["turn"] if anchor[0] == "call" else -1] += amount
    return R


def auc(scores: list[float], labels: list[int]) -> float | None:
    """The area under the ROC curve (Mann-Whitney, ties averaged); None without both classes."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(r for r, l in zip(ranks, labels) if l) - pos * (pos + 1) / 2) / (pos * neg)


PPO_DEFAULTS = {
    # 06: the policy's token budget is 3,072 (05: 2,048). The check call restates every limit and every
    # limited value once (dev mean 182 tokens, up to 337 on a five-point scan with three limits): 06's
    # reference trajectories run to 2,148 policy tokens (05's to 1,796), and the budget must hold the
    # reference itself (``check-harness`` reports ``train_reference_policy_tokens`` against it).
    "start_from": "", "steps": 2000, "prompts_per_step": 8, "max_new_tokens": 3072, "temperature": 1.0,
    "lr": 1e-5, "gamma": 1.0, "lam": 0.95, "clip_eps": 0.2, "value_clip_eps": 0.2, "inner_epochs": 4,
    "value_layer": 23, "value_lr": 9.18987e-4, "value_init_bias": -1.0, "critic_epochs": 8,
    "critic_window": 25, "critic_warmup": 25, "critic_gate_from": 40, "seed": 0, "split": "train",
    "eval_every": 25, "eval_split": "dev", "eval_cases": 120, "eval_batch": 16, "patience": 8,
    "regress_margin": 0.05, "keep_eval_checkpoints": 1, "sample_every": 5,
    # 03's stabilisers, each from a measured drift on this model (see 03): the turn-ending
    # <|im_end|> and the answer's scaffolding out of the actor's loss, Adam eps 1e-6, stop at the
    # answer's closing brace, KL 0.01 to the seed in the reward.
    "freeze_turn_ends": 1, "freeze_answer_structure": 1, "adam_eps": 1e-6, "stop_at_json": 1, "kl_coef": 0.01,
    # Tool gate (this notebook's arithmetic gate): stop once the sampled call fidelity, averaged
    # over 10 steps, falls below this -- the seed taught the calls, and PPO must not lose them.
    "tool_stop": 0.0,
    # 06: the same gate relative to the run's own start (flags --tool-stop-rel / --answer-stop-rel). A seed
    # that barely works starts below any absolute line, so the run's warm-up steps (the policy unchanged,
    # sampled at T=1) set the baseline: stop once the last 10 steps' call fidelity falls more than
    # (1 - tool_stop_rel) below the warm-up mean -- 0.9 is 05's gate for a seed that starts at 1.0, and an
    # absolute margin because 0.9 x a low start would sit inside sampling noise. answer_stop_rel is the
    # collapse rule 05's run showed (answers 0.95 -> 0.80 -> 0.31 in 15 steps, truncation 0.53): stop once
    # the last 10 steps' answered rate falls below answer_stop_rel x the warm-up mean, judged only when
    # that mean is at least 0.2 (below it there is no answer rate to lose yet). 0 turns either off.
    "tool_stop_rel": 0.0, "answer_stop_rel": 0.0,
    # 06: the unit of credit. "token": 03's GAE over the policy's tokens, lambda per token. "turn": a turn (one
    # assistant message: a block of tool calls, or the answer) is the action -- its reward is the sum of the
    # credits and KL penalties that land on its tokens, its value the critic's estimate at its first token (the
    # hidden state before the turn), the recursion runs over turns with lambda per turn, and every token of a
    # turn carries its turn's advantage. Why: from the 600 seed, lambda 0.95 per token discounted a skipped
    # conversion's bad news (a simulate error hundreds of policy tokens after the convert-or-simulate decision)
    # to nothing by the time it reached the decision, and lambda 1 per token credited every token of a turn
    # with the whole return, the needless convert calls alike; the turn is the decision, its tokens are not.
    "adv_unit": "token",
    # 06: the watchdog that catches a hung step (30 minutes, as before) is re-armed for the evaluation with its
    # own limit: 120 dev questions take half an hour with a seed that barely works (05's took minutes), and a
    # step plus the evaluation under one 30-minute timer ended every attempt at the first evaluation.
    "eval_watchdog": 7200,
    # 08: the step's own watchdog as a setting (07: 30 minutes, fixed). On Apple silicon another job on the GPU (07's
    # held-out probes beside 08's run) makes a 7-minute step 3x slower, and a killed step resumes from the last
    # evaluation.
    "step_watchdog": 1800,
    # 07: early stops read off the sampled reward -- every step's 8 episodes at T=1, so they cost nothing -- so that a
    # run that is not working ends within hours rather than at its step budget (on Apple silicon an evaluation is
    # ~100 minutes and a step ~7). The baseline is the warm-up: the seed itself, sampled on the same questions.
    #   degradation: from `reward_stop_from` actor steps on, the mean sampled reward of the last `reward_window` steps
    #     below the warm-up mean by more than `reward_stop_drop` at `reward_stop_patience` steps in a row;
    #   stall: at each actor step listed in `stall_at`, that mean not above the warm-up mean by more than
    #     `stall_margin`.
    # A 20-step mean is 160 episodes (standard error ~0.03 on this reward), so a 0.08 drop held for 5 steps is not
    # noise. 0 or "" turns a rule off.
    "reward_window": 20, "reward_stop_drop": 0.0, "reward_stop_patience": 5, "reward_stop_from": 20,
    "stall_at": "", "stall_margin": 0.0,
    # 07: take the step-0 evaluation from this run directory's eval.jsonl when one is there with the same start and
    # the same evaluation (greedy: it is the same evaluation), instead of 100 more minutes on Apple silicon.
    "reuse_eval0": 0,
    # 08: the critic. "policy" is 07's: a linear head on the policy's own layer `value_layer`, trained by Adam on every
    # policy token, and no prover. A CRITICS key is that model, frozen, read at its layer `critic_layer` before each
    # turn, with heads v and q refitted at every step on the window (`critic_window` steps; ridge penalty
    # `critic_ridge`, logistic `critic_l2`); 0 takes critic-fit's choice for that model (08_prover_critic_critic_fit.json).
    # Every turn's advantage then gains `prover_alpha` x its progress under the prover BoK(pi), K = `prover_k`.
    "critic": "policy", "critic_layer": 0, "critic_ridge": 0.0, "critic_l2": 0.0, "prover_k": 4, "prover_alpha": 0.0,
    # 08: begin after another run's first `critic_warmup` steps instead of sampling them again -- the same start, the
    # same questions (the same rng), the policy not yet updated in them: their episodes (replayed: ``replay``) fill the
    # separate critic's window, their sampled rewards are the gates' warm-up baseline, and that run's step-0
    # evaluation is this run's (when its evaluation is the same).
    "warm_from": "",
    # 08: three of "The 37 Implementation Details of Proximal Policy Optimization" (Huang et al., ICLR blog track 2022)
    # that 03-07's update did not have. `adv_norm` "batch" (or "minibatch"): the advantages minus their mean, over
    # their sd, over the tokens the actor's loss reads (#7; 03-07 used them raw, and a batch's mean was often -0.2 to
    # +0.3: every token of the batch pushed one way). `minibatches`: each inner epoch's episodes shuffled into this many
    # minibatches, an optimizer step each (#6; 03-07: one step per epoch on the whole batch, 4 steps per PPO step).
    # `target_kl`: the step's updates stop once a minibatch's approximate KL from the rollout policy, mean((r - 1) -
    # log r), passes it (the details' early stopping). approx_kl and the first minibatch's ratio (1 by construction: a
    # check) are logged either way. The defaults are 03-07's update, exactly.
    "adv_norm": "none", "minibatches": 1, "target_kl": 0.0,
    # 08: two more of the details' defaults, as settings: `lr_anneal` 1 -- the actor's learning rate decays linearly
    # from `lr` to 0 over the actor steps (03-07: constant); `max_grad_norm` -- the global gradient-norm clip (the
    # details' 0.5; 03-07: 1.0). The details' lr (2.5e-4 to 3e-4) and entropy bonus are not taken: they train small
    # networks from scratch, and here a LoRA of a pretrained 0.8B model ran away at lr 1e-5 (06).
    "lr_anneal": 0, "max_grad_norm": 1.0,
    # 08: where the rollouts and evaluations are generated. "torch": transformers' generate on the device (03-07).
    # "mlx": mlx-lm on Apple silicon (MLXPolicy / rollout_mlx), the policy's LoRA folded in before each rollout; the
    # update and every log-probability stay PyTorch's.
    "generator": "torch",
    # 08: also save the continuation state (last/) every this many steps, not only at the evaluations (0: only
    # there) -- a run on a laptop's battery loses at most this many steps to a shutdown.
    "save_every": 0,
}


def train_ppo(cfg: dict, run_dir: Path, progress=print, resume: bool = False) -> dict:
    import faulthandler
    import signal
    from collections import deque
    import torch
    device = pick_device()  # 07: CUDA, MPS or CPU (06: CUDA only)
    if cfg.get("adv_unit", "token") not in ("token", "turn"):
        raise SystemExit(f"--adv-unit must be token or turn, not {cfg['adv_unit']!r}")
    cases = load_split(cfg["split"])
    eval_cases = random.Random(0).sample(load_split(cfg["eval_split"]), cfg["eval_cases"])
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    policy, tok = load_trainable(run_dir / "last" / "adapter" if resume else HERE / cfg["start_from"])
    reference = None
    if cfg["kl_coef"]:
        reference, _ = load_trainable(HERE / cfg["start_from"])
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    suppress = tok.convert_tokens_to_ids("<|endoftext|>")
    hidden_size = policy.get_base_model().config.get_text_config().hidden_size
    head = _value_head(hidden_size, cfg["value_init_bias"]).to(device)
    actor_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(actor_params, lr=cfg["lr"], weight_decay=0.0, eps=cfg["adam_eps"])
    critic_opt = torch.optim.AdamW(head.parameters(), lr=cfg["value_lr"], weight_decay=0.0)
    window: deque = deque(maxlen=cfg["critic_window"])
    # 08: the separate critic (``critic`` a CRITICS key; cmd_ppo has resolved its layer and penalties): the frozen
    # model, its heads, and the policy's turn header for ``episode_text``. Its window holds, per step, every turn's
    # (features, return to go, the episode's success) instead of 07's per-token hidden states.
    separate = cfg.get("critic", "policy") != "policy"
    critic = heads = prefix = None
    if separate:
        critic = SeparateCritic(cfg["critic"], (cfg["critic_layer"],), device, truncate=True)
        heads = CriticHeads(cfg["critic_ridge"], cfg["critic_l2"], init_v=cfg["value_init_bias"])
        prefix = generation_prefix(tok)
    metrics_path, eval_path = run_dir / "metrics.jsonl", run_dir / "eval.jsonl"
    history: list[dict] = []
    gate = {"best": None, "best_step": None, "since_best": 0, "regress": 0, "critic_fail": 0, "step0": None,
            "reward_low": 0}
    start = 0
    prior0 = None
    if resume:
        # Everything a continuation needs, saved at the last evaluation (`last/`): a restart from the
        # adapter alone is not a continuation (`smoke_test/14`: Adam's moments and the critic's window
        # matter), so the run picks up exactly where it was.
        state = torch.load(run_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["opt"])
        critic_opt.load_state_dict(state["critic_opt"])
        window.extend(state["window"])
        if separate:
            heads.state = state.get("heads")
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        set_device_rng_state(device, state.get("device_rng"))
        gate, start = state["gate"], state["step"]
        history = [r for r in map(json.loads, metrics_path.read_text().splitlines()) if r["step"] < start]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        evals = [r for r in map(json.loads, eval_path.read_text().splitlines()) if r["step"] <= start]
        eval_path.write_text("".join(json.dumps(r) + "\n" for r in evals))
        samples = run_dir / "samples.jsonl"
        if samples.exists():
            kept = [r for r in map(json.loads, samples.read_text().splitlines()) if r["step"] < start]
            samples.write_text("".join(json.dumps(r) + "\n" for r in kept))
        progress(f"resumed {run_dir.name} at step {start} (generator {cfg.get('generator', 'torch')})")
    else:
        # 07: a step-0 evaluation this run directory already has, for ``reuse_eval0``
        if cfg.get("reuse_eval0") and eval_path.exists() and (run_dir / "config.json").exists():
            old = json.loads((run_dir / "config.json").read_text())
            prior = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
            same = all(old.get(k) == cfg.get(k) for k in ("start_from", "eval_split", "eval_cases", "eval_batch",
                                                          "max_new_tokens", "stop_at_json", "seed"))
            if prior and prior[0]["step"] == 0 and same:
                prior0 = prior[0]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        metrics_path.write_text("")
        eval_path.write_text("")
        (run_dir / "samples.jsonl").write_text("")  # 07: a fresh run keeps no samples of an earlier attempt
    pool = ToolPool(6)
    if cfg.get("warm_from") and not resume:
        # 08: the first `critic_warmup` steps taken from another run (PPO_DEFAULTS "warm_from").
        src = HERE / cfg["warm_from"]
        old = json.loads((src / "config.json").read_text())
        same_run = ("start_from", "seed", "split", "prompts_per_step", "max_new_tokens", "temperature", "stop_at_json",
                    "kl_coef", "critic_warmup")
        if not separate or any(old.get(k) != cfg.get(k) for k in same_run):
            raise SystemExit(f"--warm-from needs a separate critic and a run with the same {', '.join(same_run)}")
        n_warm = cfg["critic_warmup"]
        rows = [r for r in read_jsonl(src / "samples.jsonl") if r["step"] < n_warm]
        for s in range(n_warm):  # the same rng draws the same questions: checked, and the rng moved past them
            drawn = [rng.choice(cases)["id"] for _ in range(cfg["prompts_per_step"])]
            if [r["id"] for r in rows if r["step"] == s] != drawn:
                raise SystemExit(f"--warm-from: {src.name}'s step {s} has other questions (it must sample every step)")
        eps = replay(rows, {c["id"]: c for c in cases}, pool)
        mismatch = 0
        for s in range(n_warm):
            X, G, Y = [], [], []
            for r, ep in zip(rows, eps):
                if r["step"] != s:
                    continue
                sc = score_episode(ep)
                mismatch += abs(sc["total"] - r["reward"]) > 1e-9
                X.append(critic.features(*episode_text(ep, tok, prefix))[critic.layers[0]])
                # The return to go of each turn: no KL penalty, the policy being the reference in these steps.
                G.append(torch.tensor(turn_rewards(ep, sc), dtype=torch.float32).flip(0).cumsum(0).flip(0))
                Y.append(torch.full((len(ep.turns),), float(success(sc))))
            window.append((torch.cat(X), torch.cat(G), torch.cat(Y)))
        heads.fit(*(torch.cat([w[k] for w in window]) for k in range(3)))
        history = [dict(r, warm_from=cfg["warm_from"]) for r in read_jsonl(src / "metrics.jsonl") if r["step"] < n_warm]
        if len(history) != n_warm:
            raise SystemExit(f"--warm-from: {src.name} has {len(history)} of its first {n_warm} steps' metrics")
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        same_eval = ("start_from", "eval_split", "eval_cases", "eval_batch", "max_new_tokens", "stop_at_json", "seed")
        prior = [r for r in read_jsonl(src / "eval.jsonl") if r["step"] == 0]
        if prior and all(old.get(k) == cfg.get(k) for k in same_eval):
            prior0 = prior[0]
        start = n_warm
        progress(f"warm start from {src.name}: its first {n_warm} steps ({len(rows)} episodes replayed, {mismatch} "
                 f"rewards off the recorded ones), warm-up reward {sum(r['reward_mean'] for r in history) / n_warm:.3f}"
                 f"; critic window {sum(len(w[0]) for w in window)} turns, {critic.seconds:.0f}s of critic reads")

    def save_state(step: int) -> None:
        last = run_dir / "last"
        last.mkdir(exist_ok=True)
        policy.save_pretrained(last / "adapter")
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                    "window": list(window), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "device_rng": device_rng_state(device), "gate": dict(gate), "step": step,
                    "heads": heads.state if separate else None}, last / "state.pt")

    # 08: the generator (PPO_DEFAULTS "generator"): transformers' generate, or mlx-lm with the policy's LoRA folded in
    gen = MLXPolicy() if cfg.get("generator", "torch") == "mlx" else None

    def generate_episodes(batch: list[dict], temperature: float) -> list[Episode]:
        if gen is None:
            return rollout(policy, tok, batch, pool, temperature=temperature, max_new_tokens=cfg["max_new_tokens"],
                           stop_at_json=bool(cfg["stop_at_json"]))
        import mlx.core as mx
        gen.sync(policy)
        mx.random.seed(int(torch.randint(2 ** 31 - 1, ()).item()))  # MLX's sampling drawn from the run's torch rng
        return rollout_mlx(gen, tok, batch, pool, temperature=temperature, max_new_tokens=cfg["max_new_tokens"],
                           stop_at_json=bool(cfg["stop_at_json"]))

    def critic_window_stats(n=25):
        # 08: not the warm start's steps (``warm_from``): their value_ev is the other run's critic's
        rows = [r for r in history[-n:] if r.get("value_ev") is not None and r.get("value_ev_position") is not None
                and not r.get("warm_from")]
        if not rows:
            return {"critic_residual_median": None, "critic_ahead_frac": None}
        res = sorted(r["value_ev"] - r["value_ev_position"] for r in rows)
        mid = len(res) // 2
        med = res[mid] if len(res) % 2 else (res[mid - 1] + res[mid]) / 2
        return {"critic_residual_median": round(med, 6), "critic_ahead_frac": round(sum(x > 0 for x in res) / len(res), 4)}

    def run_eval(step: int) -> str | None:
        policy.eval()
        state = torch.get_rng_state(), device_rng_state(device)
        eps, scs = [], []
        for b in range(0, len(eval_cases), cfg["eval_batch"]):
            got = generate_episodes(eval_cases[b: b + cfg["eval_batch"]], 0.0)
            eps += got
            scs += [score_episode(e) for e in got]
        torch.set_rng_state(state[0])
        set_device_rng_state(device, state[1])
        release_device_cache()
        o = summarise(eps, scs)["overall"]
        rec = {"step": step, **{k: o[k] for k in ("reward", "decision_correct", "evidence", "answer_found",
                                                  "calls_per_episode", "calls_errored", "truncated", "policy_tokens")},
               "components": o["components"], "recovery": o["recovery"], **critic_window_stats()}
        reason = None
        if step == 0:
            gate["step0"] = rec["reward"]
        if step and cfg["keep_eval_checkpoints"]:
            policy.save_pretrained(run_dir / f"adapter-step{step}")
        if gate["best"] is None or rec["reward"] > gate["best"]:
            gate.update(best=rec["reward"], best_step=step, since_best=0)
            policy.save_pretrained(run_dir / "adapter-best")
            (run_dir / "best.json").write_text(json.dumps(rec, indent=2) + "\n")
        else:
            gate["since_best"] += 1
            if gate["since_best"] >= cfg["patience"]:
                reason = f"converged: {cfg['patience']} evaluations without beating {gate['best']:.4f} (step {gate['best_step']})"
        gate["regress"] = gate["regress"] + 1 if rec["reward"] < gate["step0"] - cfg["regress_margin"] else 0
        if gate["regress"] >= 2:
            reason = f"no-regression gate: held-out reward {rec['reward']:.4f} below step 0 twice"
        if step >= cfg["critic_gate_from"] and rec["critic_residual_median"] is not None:
            gate["critic_fail"] = gate["critic_fail"] + 1 if rec["critic_residual_median"] <= 0 else 0
            if gate["critic_fail"] >= 2:
                reason = "critic gate: value_ev not above the position clock at two evaluations"
        rec["gate"] = dict(gate)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if step:
            save_state(step)
        crit = rec["critic_residual_median"]
        rg = rec["recovery"]["recovered_given_error"]
        progress(f"    eval @{step:<5} reward {rec['reward']:.4f} decision {rec['decision_correct']:.3f} "
                 f"evidence {rec['evidence']:.3f} answer {rec['answer_found']:.3f} calls {rec['calls_per_episode']:.2f} "
                 f"erred {rec['recovery']['sim_erred']:.3f} recovered {'-' if rg is None else f'{rg:.3f}'} "
                 f"critic {'-' if crit is None else f'{crit:+.4f}'} | best {gate['best']:.4f}@{gate['best_step']}")
        return reason

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    progress(f"{MODEL} from {cfg['start_from']} | value layer {cfg['value_layer']} lr {cfg['value_lr']:.3g} "
             f"bias {cfg['value_init_bias']:.4f} | kl {cfg['kl_coef']} | warm-up {cfg['critic_warmup']} | "
             f"lambda {cfg['lam']} per {cfg.get('adv_unit', 'token')}")
    if separate:
        progress(f"critic {cfg['critic']} ({CRITICS[cfg['critic']]}) layer {cfg['critic_layer']} of {critic.depth}, "
                 f"ridge {cfg['critic_ridge']:g} l2 {cfg['critic_l2']:g}, window {cfg['critic_window']} steps | prover "
                 f"Bo{cfg['prover_k']}(pi), alpha {cfg['prover_alpha']:g}")

    def reuse_eval(rec: dict) -> None:
        """07: the step-0 evaluation taken from an earlier attempt of this run (``reuse_eval0``); 08: or from the run
        this one warm-starts from (``warm_from``)."""
        gate.update(step0=rec["reward"], best=rec["reward"], best_step=0, since_best=0)
        policy.save_pretrained(run_dir / "adapter-best")
        rec = {k: v for k, v in rec.items() if k != "gate"} | {"reused": True, "gate": dict(gate)}
        (run_dir / "best.json").write_text(json.dumps(rec, indent=2) + "\n")
        with eval_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        progress(f"    eval @0     reward {rec['reward']:.4f} decision {rec['decision_correct']:.3f} (taken from "
                 f"{cfg.get('warm_from') or 'the earlier attempt'}'s step-0 evaluation: same start, same greedy evaluation)")

    stop = None
    if cfg["eval_every"] and not resume:
        if prior0 is not None:
            reuse_eval(prior0)
        else:
            stop = run_eval(0)
    step = start
    try:
        while stop is None and step < cfg["steps"]:
            faulthandler.dump_traceback_later(cfg.get("step_watchdog", 1800), exit=True)
            t0 = time.perf_counter()
            batch_cases = [rng.choice(cases) for _ in range(cfg["prompts_per_step"])]
            policy.eval()
            episodes = generate_episodes(batch_cases, cfg["temperature"])
            gen_s = time.perf_counter() - t0
            B = build_batch(episodes, tok, device)
            seqs, attn, span, index, mask, rewards = B["seqs"], B["attn"], B["span"], B["index"], B["mask"], B["rewards"]
            with torch.no_grad():
                old_lp, hid, mass = [], [], []
                for i in range(len(seqs)):
                    lp, h, m = forward_policy(policy, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1],
                                              suppress=suppress, hidden_layer=cfg["value_layer"])
                    old_lp.append(lp), hid.append(h), mass.append(m)
                old_lp, hidden, mass = torch.cat(old_lp), torch.cat(hid), torch.cat(mass)
                kl = torch.zeros_like(old_lp)
                if reference is not None:
                    ref_lp = torch.cat([forward_policy(reference, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1],
                                                       suppress=suppress)[0] for i in range(len(seqs))])
                    kl = (old_lp - ref_lp) * mask
                    rewards = rewards - cfg["kl_coef"] * kl
                if separate:
                    # 08: the separate critic's reads of the states before each turn, with the heads fitted on the
                    # window so far (this step's episodes are not in it: its values are out of sample).
                    feats = [critic.features(*episode_text(ep, tok, prefix))[critic.layers[0]] for ep in episodes]
                    v_turn = [heads.value(f) for f in feats]
                    q_turn = [heads.success(f) for f in feats]
                    old_values = per_turn_tokens(v_turn, B["turn"], mask)
                else:
                    old_values = _values(head, hidden) * mask
                if cfg["adv_unit"] == "turn":
                    adv, returns = gae_turn(rewards, old_values, mask, B["turn"], gamma=cfg["gamma"], lam=cfg["lam"])
                else:
                    adv, returns = gae(rewards, old_values, mask, gamma=cfg["gamma"], lam=cfg["lam"])
                prog = None
                if separate:
                    # 08: the prover's progress per turn (``prover_progress``), added to the turn's advantage after
                    # GAE -- the paper's per-step Q^pi + alpha A^mu: each turn gains its own A^mu, not a sum to go.
                    ys = [float(success(s)) for s in B["scores"]]
                    prog_turn = [prover_progress(q, y_, cfg["prover_k"]) for q, y_ in zip(q_turn, ys)]
                    prog = per_turn_tokens(prog_turn, B["turn"], mask)
                    if cfg["prover_alpha"]:
                        adv = adv + cfg["prover_alpha"] * prog
            sel = mask > 0
            cstats = {}
            if separate:
                # 08: this step's turns join the window -- (features, return to go incl. the KL penalties, success) --
                # and both heads are refitted on it from scratch.
                G_turn = []
                for i, ep in enumerate(episodes):
                    on = sel[i]
                    R = torch.zeros(len(ep.turns), dtype=torch.float64).index_add_(
                        0, B["turn"][i][on].cpu(), rewards[i][on].cpu().double())  # on the CPU: MPS has no float64
                    G_turn.append(R.flip(0).cumsum(0).flip(0).float())
                Y_turn = [torch.full((len(g),), y_) for g, y_ in zip(G_turn, ys)]
                window.append((torch.cat(feats), torch.cat(G_turn), torch.cat(Y_turn)))
                cstats = separate_critic_stats(v_turn, q_turn, G_turn, ys, prog_turn, cfg["prover_alpha"])
                heads.fit(*(torch.cat([w[k] for w in window]) for k in range(3)))
            else:
                window.append((hidden[sel].cpu(), old_values[sel].cpu(), returns[sel].cpu()))
                fh_, fv_, fr_ = (torch.cat([w[k] for w in window]).to(device) for k in range(3))
                for _ in range(cfg["critic_epochs"]):
                    critic_opt.zero_grad(set_to_none=True)
                    vl, cstats = value_loss(_values(head, fh_), fv_, fr_, clip_eps=cfg["value_clip_eps"])
                    vl.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    critic_opt.step()
                del fh_, fv_, fr_
            actor_mask = mask * (1 - B["turn_end"]) if cfg["freeze_turn_ends"] else mask
            if cfg["freeze_answer_structure"]:
                actor_mask = actor_mask * (1 - B["answer"] + B["content"])
            warming = step < cfg["critic_warmup"]
            astats, grad_norm = {}, torch.zeros(())
            # 08: three of the 37 PPO implementation details (PPO_DEFAULTS: adv_norm, minibatches, target_kl). With
            # their defaults this is 03-07's update exactly: one step per epoch on the whole batch, in episode order.
            if cfg.get("adv_norm") == "batch":
                adv = normalized_advantages(adv, actor_mask)
            n_mb = max(1, min(int(cfg.get("minibatches", 1)), len(seqs)))
            if cfg.get("lr_anneal") and not warming:  # computed from the step: a resumed run lands on the same rate
                frac = 1.0 - (step - cfg["critic_warmup"]) / max(1, cfg["steps"] - cfg["critic_warmup"])
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * max(frac, 0.0)
            updates, kl_stop, approx_kls, ratio_first = 0, False, [], None
            policy.train()
            for epoch in range(0 if warming else cfg["inner_epochs"]):
                order = list(range(len(seqs)))
                if n_mb > 1:  # its own rng: the run's rng draws the questions (the same as 07's at every step)
                    random.Random(f"{cfg['seed']}:{step}:{epoch}").shuffle(order)
                ep_stats = defaultdict(float)
                for mb in (order[k::n_mb] for k in range(n_mb)):
                    opt.zero_grad(set_to_none=True)
                    adv_mb = adv
                    if cfg.get("adv_norm") == "minibatch":
                        sel = torch.zeros_like(actor_mask)
                        sel[mb] = actor_mask[mb]
                        adv_mb = normalized_advantages(adv, sel)
                    mb_kl, mb_ratio = 0.0, 0.0
                    for i in mb:
                        lp, _, _ = forward_policy(policy, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1], suppress=suppress)
                        loss, st_ = ppo_actor_loss(lp, old_lp[i:i + 1], adv_mb[i:i + 1], actor_mask[i:i + 1],
                                                   clip_eps=cfg["clip_eps"])
                        (loss / len(mb)).backward()
                        for k, v in st_.items():
                            ep_stats[k] += v / len(seqs)
                        mb_kl += st_["approx_kl"] / len(mb)
                        mb_ratio += st_["ratio_mean"] / len(mb)
                    grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, float(cfg.get("max_grad_norm", 1.0)))
                    opt.step()
                    updates += 1
                    approx_kls.append(mb_kl)
                    if ratio_first is None:
                        ratio_first = mb_ratio  # the policy not yet moved this step: 1 up to numerics
                    if cfg.get("target_kl") and mb_kl > cfg["target_kl"]:
                        kl_stop = True
                        break
                astats = dict(ep_stats)
                if kl_stop:
                    break
            if updates:
                astats.update(updates=updates, kl_stop=kl_stop, approx_kl_last=approx_kls[-1],
                              approx_kl_max=max(approx_kls), ratio_first=ratio_first)
            release_device_cache()
            scs = B["scores"]
            active = mask.sum().clamp(min=1)
            mean_adv = float((adv * mask).sum() / active)
            ans = B["answer"] * mask
            rec = {
                "step": step, "critic_warmup": warming,
                "reward_mean": sum(s["total"] for s in scs) / len(scs),
                "calls_fidelity": sum(s["components"]["calls"] for s in scs) / len(scs),
                "decision_correct": sum(s["decision_correct"] for s in scs) / len(scs),
                "evidence": sum(s["evidence"] for s in scs) / len(scs),
                "answered": float(ans.sum(dim=-1).gt(0).float().mean()),
                "calls_mean": sum(len(e.calls) for e in episodes) / len(episodes),
                "truncated": sum(e.truncated for e in episodes) / len(episodes),
                "policy_tokens_mean": float(mask.sum(dim=-1).float().mean()),
                "episode_tokens_mean": sum(len(e.prompt_ids) + len(e.ids) for e in episodes) / len(episodes),
                "kl_mean": round(float(kl.sum() / active), 6),
                "kl_answer": round(float((kl * ans).sum() / ans.sum().clamp(min=1)), 6),
                # 08: the rollout policy's entropy per token, estimated free: the tokens were sampled from it at T=1, so
                # the mean of -log p over them is its entropy in expectation (over the tokens the actor's loss reads)
                "entropy_sampled": round(float((-old_lp * actor_mask).sum() / actor_mask.sum().clamp(min=1)), 6),
                "suppressed_mass_max": float((mass * mask).max()),
                "adv_mean": round(mean_adv, 6),
                "adv_std": round(float((((adv - mean_adv) * mask) ** 2).sum().div(active).sqrt()), 6),
                **critic_fit(old_values, returns, mask),
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in astats.items()},
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in cstats.items()},
                "grad_norm": float(grad_norm), "lr": opt.param_groups[0]["lr"], "gen_seconds": round(gen_s, 1),
                "step_seconds": round(time.perf_counter() - t0, 1), "solves": pool.solved,
                "generator": cfg.get("generator", "torch"),  # 08: a resumed run may switch it
                # 07: what the step's episodes did about their erroring simulate calls (``recovery_block``)
                **{f"rec_{k}": (round(v, 6) if isinstance(v, float) else v)
                   for k, v in recovery_block([recovery_stats(e.calls) for e in episodes], scs).items()},
            }
            history.append(rec)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if step % cfg["sample_every"] == 0:
                with (run_dir / "samples.jsonl").open("a") as fh:
                    for ep, sc in zip(episodes, scs):
                        fh.write(json.dumps({"step": step, **ep.to_json(), "reward": sc["total"]}) + "\n")
            ev, clock = rec["value_ev"], rec["value_ev_position"]
            rg = rec["rec_recovered_given_error"]
            progress(f"  step {step:>5} reward {rec['reward_mean']:.4f} calls {rec['calls_fidelity']:.3f} "
                     f"dec {rec['decision_correct']:.2f} evid {rec['evidence']:.2f} V {rec.get('value_mean', 0):.3f} "
                     f"ev {'undef' if ev is None else f'{ev:+.3f}'} clock {'undef' if clock is None else f'{clock:+.3f}'} "
                     f"adv {rec['adv_mean']:+.4f} kl {rec['kl_mean']:.4f} tok {rec['policy_tokens_mean']:.0f}/"
                     f"{rec['episode_tokens_mean']:.0f} err {rec['rec_sim_erred']:.2f} "
                     f"recov {'-' if rg is None else f'{rg:.2f}'} {rec['step_seconds']:.0f}s")
            if separate:
                cev = rec["critic_ev_turn"]
                progress(f"         critic ev(turn) {'undef' if cev is None else f'{cev:+.3f}'} q {rec['q_mean']:.2f} "
                         f"(q0 {rec['q0_mean']:.2f}, success {rec['success_rate']:.2f}, brier {rec['q_brier']:.3f}) "
                         f"A^mu {rec['prog_mean']:+.3f} sd {rec['prog_std']:.3f}")
            if rec.get("updates") and (cfg.get("adv_norm", "none") != "none" or int(cfg.get("minibatches", 1)) > 1
                                       or cfg.get("target_kl")):
                progress(f"         update: {rec['updates']} optimizer steps{' (stopped: target_kl)' if rec['kl_stop'] else ''}, "
                         f"approx_kl last {rec['approx_kl_last']:.4f} max {rec['approx_kl_max']:.4f}, first ratio "
                         f"{rec['ratio_first']:.4f}, clip frac {rec.get('clip_frac', 0):.3f}, entropy {rec['entropy_sampled']:.3f}")
            step += 1
            if cfg["eval_every"] and step % cfg["eval_every"] == 0:
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(cfg["eval_watchdog"], exit=True)
                stop = run_eval(step)
                faulthandler.cancel_dump_traceback_later()
            if stop is None and cfg["tool_stop"] and len(history) >= 10:
                recent = sum(r["calls_fidelity"] for r in history[-10:]) / 10
                if recent < cfg["tool_stop"]:
                    stop = f"tool gate: sampled call fidelity over the last 10 steps {recent:.3f} < {cfg['tool_stop']}"
            if stop is None and (cfg["tool_stop_rel"] or cfg["answer_stop_rel"]) and len(history) >= 10:
                base = [r for r in history if r.get("critic_warmup")] or history[:10]
                start_fid = sum(r["calls_fidelity"] for r in base) / len(base)
                start_ans = sum(r["answered"] for r in base) / len(base)
                recent_fid = sum(r["calls_fidelity"] for r in history[-10:]) / 10
                recent_ans = sum(r["answered"] for r in history[-10:]) / 10
                if cfg["tool_stop_rel"] and recent_fid < start_fid - (1 - cfg["tool_stop_rel"]):
                    stop = (f"tool gate (relative): sampled call fidelity over the last 10 steps {recent_fid:.3f} < "
                            f"the warm-up mean {start_fid:.3f} - {1 - cfg['tool_stop_rel']:.2f}")
                elif cfg["answer_stop_rel"] and start_ans >= 0.2 and recent_ans < cfg["answer_stop_rel"] * start_ans:
                    stop = (f"answer collapse: sampled answered rate over the last 10 steps {recent_ans:.3f} < "
                            f"{cfg['answer_stop_rel']} x the warm-up mean {start_ans:.3f}")
            # 07: the reward gates (PPO_DEFAULTS): degradation and stall, against the warm-up's sampled reward.
            actor = [r for r in history if not r.get("critic_warmup")]
            base = [r["reward_mean"] for r in history if r.get("critic_warmup")]
            w = cfg["reward_window"]
            if stop is None and base and w and len(actor) >= w:
                recent = sum(r["reward_mean"] for r in actor[-w:]) / w
                base_mean = sum(base) / len(base)
                if cfg["reward_stop_drop"] and len(actor) >= cfg["reward_stop_from"]:
                    low = recent < base_mean - cfg["reward_stop_drop"]
                    gate["reward_low"] = gate.get("reward_low", 0) + 1 if low else 0
                    if gate["reward_low"] >= cfg["reward_stop_patience"]:
                        stop = (f"reward gate: the last {w} steps' sampled reward {recent:.3f} below the warm-up's "
                                f"{base_mean:.3f} - {cfg['reward_stop_drop']} at {gate['reward_low']} steps in a row")
                stalls = {int(x) for x in str(cfg["stall_at"]).split(",") if x.strip()}
                if stop is None and len(actor) in stalls and recent <= base_mean + cfg["stall_margin"]:
                    stop = (f"stall gate: after {len(actor)} actor steps the last {w} steps' sampled reward {recent:.3f} "
                            f"is not above the warm-up's {base_mean:.3f} + {cfg['stall_margin']}")
                if len(actor) in stalls or len(actor) % 10 == 0:
                    progress(f"    reward gates @ actor step {len(actor)}: last {w} steps {recent:.3f} against the warm-up's "
                             f"{base_mean:.3f} (degradation count {gate.get('reward_low', 0)})")
            # 08: the continuation state between evaluations too (PPO_DEFAULTS "save_every"), after the step's gates
            if stop is None and cfg.get("save_every") and step % cfg["save_every"] == 0:
                save_state(step)
    finally:
        faulthandler.cancel_dump_traceback_later()
        pool.close()
    policy.save_pretrained(run_dir / "adapter")
    torch.save(heads.state if separate else head.state_dict(), run_dir / ("critic_heads.pt" if separate else "value_head.pt"))
    ending = {"stopped_at": step, "reason": stop or f"step budget {cfg['steps']} reached",
              "best_step": gate["best_step"], "best_reward": gate["best"]}
    (run_dir / "ending.json").write_text(json.dumps(ending, indent=2) + "\n")
    progress(f"END {ending}")
    return ending



def episode_report(path: Path, split: str) -> dict:
    """What a probe's episodes did, by family and by unit level, and which answers they said."""
    cases = {c["id"]: c for c in load_split(split)}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        c = cases[r["id"]]
        by[("family", c["family"])].append((r, c))
        by[("level", c["level"])].append((r, c))
        by[("all", "")].append((r, c))
    out = {}
    for key, pairs in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n = len(pairs)
        calls = [x for r, _ in pairs for x in r["calls"]]
        said = Counter()
        for r, c in pairs:
            obj = extract_answer(r["final_text"])[0]
            if obj is None:
                said["(no answer)"] += 1
            else:
                d = obj.get("decision")
                if _none_like(d):
                    said["none"] += 1
                elif c["decision_kind"] == "number":
                    x = parse_number(d)
                    users = c["scan"]["users"]
                    hit = [i for i, u in enumerate(users) if x is not None and abs(x - u) <= 0.002 * abs(u)]
                    said[f"{hit[0] + 1}/{len(users)}" if hit else "other"] += 1
                else:
                    said[str(d)[:12]] += 1
        out[f"{key[0]}:{key[1]}"] = {
            "n": n, "reward": sum(r["score"]["total"] for r, _ in pairs) / n,
            "answer_found": sum(extract_answer(r["final_text"])[0] is not None for r, _ in pairs) / n,
            "format": sum(r["score"]["components"]["format"] for r, _ in pairs) / n,
            "first_call_right_tool": sum(next((x["name"] for x in r["calls"] if x["name"] not in LOCAL_TOOLS), None) == c["tool"]
                                         for r, c in pairs) / n,
            "calls_errored": sum(x["error"] for x in calls) / max(1, len(calls)),
            "calls_fidelity": sum(r["score"]["components"]["calls"] for r, _ in pairs) / n,
            "evidence": sum(r["score"]["evidence"] for r, _ in pairs) / n,
            "decision_correct": sum(r["score"]["decision_correct"] for r, _ in pairs) / n,
            "values": sum(r["score"]["components"].get("values", 0) for r, _ in pairs) / n,
            "truncated": sum(r["truncated"] for r, _ in pairs) / n,
            "unanswered": sum(r["unanswered"] > 0 for r, _ in pairs) / n,
            "said": dict(said.most_common()),
        }
    return out

# ---------------------------------------------------------------------------------------------
# The seed's aliveness (06): has each behaviour the task needs been sampled at least once?
# ---------------------------------------------------------------------------------------------

LABELS = ["a tool call parsed", "a simulate call of the question's tool, ran", "a convert_units call, ran", "a check_limits call, ran",
          "the proving simulations", "a comparison line with a verdict", "the aggregation line (scan questions)",
          "an answer JSON parsed", "the decision right",
          # 07: the behaviour 06's seeds never showed. Its rate is over the episodes with an erroring simulate call.
          "an erroring simulate call, then a clean one"]
ITEM_ANY = re.compile(r"[\w.]+ -?[\d.]+(?:e-?\d+)? (?:meets|breaks) (?:<=|>=) -?[\d.]+(?:e-?\d+)?")


def labels_of(rows: list[dict], cases: dict[str, dict], scored: bool = True) -> dict:
    """Per behaviour label, the episodes it appears in; per (family, answer class), right / sampled.
    Alive: every label at least once and every one of the 16 classes right at least once -- the seed rule:
    sampled once is learned, the seed need only barely work, and PPO is asked for the rest. scored=False for
    PPO's training samples, which carry no score (their proving simulations and decisions come from the
    per-step metrics). 07: the recovery label's rate is over the episodes that erred, not all of them."""
    hit = Counter(); cls = defaultdict(lambda: [0, 0]); scans = 0; erred = 0
    for r in rows:
        c = cases[r["id"]]; calls = r["calls"]; head = r["final_text"].split("{")[0]
        hit[LABELS[0]] += bool(calls)
        hit[LABELS[1]] += any(x["name"] == c["tool"] and not x["error"] for x in calls)
        hit[LABELS[2]] += any(x["name"] == "convert_units" and not x["error"] for x in calls)
        hit[LABELS[3]] += any(x["name"] == "check_limits" and not x["error"] for x in calls)
        hit[LABELS[5]] += bool(ITEM_ANY.search(head))
        if c["decision_kind"] == "number":
            scans += 1; hit[LABELS[6]] += "passing:" in head
        hit[LABELS[7]] += extract_answer(r["final_text"])[0] is not None
        rs = recovery_stats(calls)
        erred += rs["erred"]; hit[LABELS[9]] += rs["recovered"]
        if scored:
            hit[LABELS[4]] += bool(r["score"]["evidence"]); hit[LABELS[8]] += bool(r["score"]["decision_correct"])
            k = (c["family"], decision_class(c) or c["gold"]["decision"]); cls[k][1] += 1; cls[k][0] += bool(r["score"]["decision_correct"])
    n = len(rows)
    rate = {k: hit[k] / max(1, scans if k == LABELS[6] else erred if k == LABELS[9] else n) for k in LABELS}
    alive = all(hit[k] > 0 for k in LABELS) and len(cls) >= 16 and all(v[0] > 0 for v in cls.values())
    return {"n": n, "hit": hit, "rate": rate, "classes": cls, "alive": alive, "erred": erred}


def cmd_alive(args) -> None:
    """Judge a dev probe's episodes: the labels' table, then ALIVE or NOT YET ALIVE (exit status 1)."""
    path = Path(args.episodes)
    cases = {c["id"]: c for c in load_split(args.split)}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    L = labels_of(rows, cases)
    print(f"{path}: {L['n']} episodes")
    print(f"   {'label':44} {'episodes':>9} {'rate':>6}")
    for k in LABELS:
        print(f"   {k:44} {L['hit'][k]:>9} {L['rate'][k]:>6.3f}{'' if L['hit'][k] else '  DEAD'}")
    print(f"   (the recovery rate is over the {L['erred']} episodes with an erroring simulate call)")
    dead = [f"{f}:{c}" for (f, c), v in sorted(L["classes"].items()) if v[0] == 0]
    print(f"   answer classes right at least once: {sum(1 for v in L['classes'].values() if v[0] > 0)}/{len(L['classes'])}"
          + (f"  DEAD: {dead}" if dead else ""))
    print("ALIVE" if L["alive"] else "NOT YET ALIVE")
    sys.exit(0 if L["alive"] else 1)


# ---------------------------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------------------------

def cmd_dump_tools(_args) -> None:
    import asyncio
    host = _host()
    listed = asyncio.run(host.server.mcp.list_tools())
    decls = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.inputSchema}}
             for t in listed if t.name in TOOLS] + [CONVERT_DECL, COMPARE_DECL]
    TOOLS_JSON.write_text(json.dumps(decls, indent=1) + "\n")
    print(f"wrote {TOOLS_JSON.name}: {[d['function']['name'] for d in decls]}")


def _assert_defaults() -> None:
    sys.path.insert(0, str(MCP_DIR))
    import ro_model
    import swro_model
    assert ro_model.DEFAULTS == DEFAULTS[RO], "ro_model.DEFAULTS changed"
    assert swro_model.DEFAULTS == DEFAULTS[SWRO], "swro_model.DEFAULTS changed"


def cmd_generate(args) -> None:
    """07 does not generate: its questions are 06's, read from 06's data directory."""
    raise SystemExit(f"07's questions are 06's, read in place from {QDATA.name}/ "
                     "(06_direction_and_compare.py generate draws them)")


def decision_label(case: dict) -> str:
    """The gold decision as a class: pass/fail, a label, or a scan position like "2/4"."""
    d = case["gold"]["decision"]
    if d is None:
        return "none"
    if case["decision_kind"] == "number":
        users = case["scan"]["users"]
        return f"{users.index(d) + 1}/{len(users)}"
    return str(d)


def load_split(split: str) -> list[dict]:
    """06's questions of a split (07 reads them in place)."""
    return [json.loads(line) for line in (QDATA / f"{split}.jsonl").read_text().splitlines()]


def sentence_bank(case: dict, con: dict) -> str:
    """Which bank the sentence stating one limit came from: "shared" (05's two or three, drawn by
    train, dev and test), "extra" (06's train-only ones) or "holdout" (holdout_shift's unseen one)."""
    key = (con["metric"], con["op"])
    for bank, templates in (("shared", CONSTRAINTS[key][0]), ("extra", CONSTRAINTS_TRAIN_EXTRA[key]),
                            ("holdout", CONSTRAINTS[key][1])):
        if any(t.format(v=con["text"]) in case["prompt"] for t in templates):
            return bank
    raise ValueError(f"{case['id']}: no known sentence states {key}")


def swap_sentences(case: dict, to: str) -> dict:
    """The case with every limit restated: ``to="shared"`` puts 05's first sentence for that limit
    where the prompt's sentence was, ``to="holdout"`` holdout_shift's unseen one. The numbers, the
    units and every other sentence stay. A diagnostic of the sentence, never a benchmark."""
    prompt = case["prompt"]
    for con in case["constraints"]:
        key = (con["metric"], con["op"])
        current = next(s for t in CONSTRAINTS[key][0] + CONSTRAINTS_TRAIN_EXTRA[key] + CONSTRAINTS[key][1]
                       for s in [t.format(v=con["text"])] if s in prompt)
        target = (CONSTRAINTS[key][0][0] if to == "shared" else CONSTRAINTS[key][1][0]).format(v=con["text"])
        assert prompt.count(current) == 1, (case["id"], current)
        prompt = prompt.replace(current, target)
    return dict(case, prompt=prompt)


def cmd_check(args) -> None:
    """06's checks of the questions (07 reads them unchanged: the deployed prompt, duplicates, the reward's
    sanity and null baselines, the sentences per split), and what 07's recovery episodes slipped, how the
    simulator answered, and what the fix costs in reward."""
    report: dict[str, Any] = {}
    core = AGENT_CORE.read_text()
    m = re.search(r"SYSTEM_MESSAGE = \((.*?)\n\)", core, re.S)
    deployed = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))) if m else None
    report["system_message_matches_agent"] = deployed == DEPLOYED_SYSTEM_MESSAGE
    report["questions"] = str(QDATA.relative_to(HERE))
    splits = {s: load_split(s) for s in SPLITS if (QDATA / f"{s}.jsonl").exists()}
    prompts = Counter(c["prompt"] for cs in splits.values() for c in cs)
    report["duplicate_prompts"] = sum(v - 1 for v in prompts.values() if v > 1)
    for split, cases in splits.items():
        fam = defaultdict(list)
        for c in cases:
            fam[c["family"]].append(c)
        rows = {}
        for f, cs in sorted(fam.items()):
            b = [baselines(c) for c in cs]
            dec = Counter(decision_label(c) for c in cs)
            rows[f] = {"n": len(cs), "levels": dict(Counter(c["level"] for c in cs)),
                       "optimal_calls": dict(Counter(c["optimal_calls"] for c in cs)),
                       "decisions": dict(dec.most_common()),
                       "baselines": {k: round(sum(x[k] for x in b) / len(b), 4) for k in b[0]}}
        report[split] = rows
    report["constraint_sentences"] = {split: dict(Counter(sentence_bank(c, con) for c in cases for con in c["constraints"]))
                                      for split, cases in splits.items()}
    # 07: the recovery episodes, per split.
    rec = {}
    for split, cases in splits.items():
        path = recovery_path(split)
        if not path.exists():
            continue
        eps = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {c["id"]: c for c in cases}
        slips = [e["slip"] for e in eps]
        with_input = sum(bool(slip_params(c)) for c in cases)
        rec[split] = {
            "questions": len(cases), "with an input converted alone": with_input, "recovery_episodes": len(eps),
            "by_family": dict(Counter(by_id[e["id"]]["family"] for e in eps).most_common()),
            "slipped_input": dict(Counter(f"{s['param']} [{s['unit']}]" for s in slips).most_common()),
            "slipped_unit_level": dict(Counter(s["level"] for s in slips)),
            "error_kind": dict(Counter(s["error_kind"] for s in slips).most_common()),
            "inputs_tried_before_one_erred": dict(Counter(s["tried"] for s in slips)),
            "reward_min": round(min(e["reward"] for e in eps), 4) if eps else None,
            "reward_mean": round(sum(e["reward"] for e in eps) / len(eps), 4) if eps else None,
            "reference_reward_mean": round(sum(e["reference_reward"] for e in eps) / len(eps), 4) if eps else None,
            "example_errors": {k: next(s["error"] for s in slips if s["error_kind"] == k)[:240]
                               for k in sorted({s["error_kind"] for s in slips})},
            "example_fix_lines": [next(t["text"] for t in e["turns"] if t.get("text")) for e in eps[:4]],
        }
    report["recovery"] = rec
    if args.decontam:
        report["decontamination"] = decontamination([c for cs in splits.values() for c in cs])
    out = HERE / f"{PREFIX}_checks.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


def cmd_elasticity(args) -> None:
    """Worst relative output change when one stated input is off by +-ARG_TOL (justifies MARGIN)."""
    host = _host()
    worst: dict[str, dict[str, dict[str, float]]] = {RO: defaultdict(dict), SWRO: defaultdict(dict)}
    scenarios = Counter()
    rng = random.Random(7)
    while min(scenarios[RO], scenarios[SWRO]) < args.n:
        tool = RO if scenarios[RO] <= scenarios[SWRO] else SWRO
        g = Gen(rng, "dev", 0, host)
        stated = g.ro_base()[0] if tool == RO else swro_base(g)[0]
        try:
            base = g.solve(tool, stated)["metrics"]
        except Retry:
            continue
        scenarios[tool] += 1
        for key, v in stated.items():
            if not isinstance(v, float):
                continue
            for sign in (1, -1):
                try:
                    m = g.solve(tool, {**stated, key: v * (1 + sign * ARG_TOL)})["metrics"]
                except Retry:
                    continue
                for path, b in base.items():
                    d = abs(m[path] / b - 1)
                    worst[tool][key][path] = max(worst[tool][key].get(path, 0.0), d)
    out = {"arg_tol": ARG_TOL, "scenarios": dict(scenarios),
           "worst_by_input": {t: {k: dict(v) for k, v in w.items()} for t, w in worst.items()},
           "worst_by_metric": {t: {p: max(v.get(p, 0.0) for v in w.values()) for p in METRICS[t]}
                               for t, w in worst.items()},
           "margin": {p: MARGIN[p] for t in TOOLS for p in METRICS[t]}}
    (HERE / f"{PREFIX}_elasticity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["worst_by_metric"], indent=1))



def _template_check(tok, case: dict, ref: dict, prefix: str, im_end: int) -> tuple[bool, bool]:
    """(the calls parse back to what was written, the episode built by token concatenation is the chat
    template's own rendering) -- for a blind reference episode or a recovery episode (07: a turn's own line
    goes in the assistant message's content, before its tool calls)."""
    ids = tok(render_prompt(tok, case), add_special_tokens=False).input_ids
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case["prompt"]}]
    ok = True
    for turn in ref["turns"]:
        text = turn_text(turn)
        parsed = parse_calls(text)
        ok &= [(p["name"], p["arguments"]) for p in parsed] == [(x["name"], x["arguments"]) for x in turn["calls"]]
        ids += tok(text, add_special_tokens=False).input_ids + [im_end]
        ids += tok(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
        msgs.append({"role": "assistant", "content": turn.get("text", ""), "tool_calls": [
            {"type": "function", "function": {"name": p["name"], "arguments": p["arguments"]}} for p in parsed]})
        msgs += [{"role": "tool", "content": r} for r in turn["results"]]
    answer = ref["answer_reasoned"]
    ids += tok(answer, add_special_tokens=False).input_ids + [im_end]
    msgs.append({"role": "assistant", "content": answer})
    text = tok.apply_chat_template(msgs, tools=_tool_decls(), tokenize=False, **TEMPLATE_KWARGS)
    return ok, ids == tok(text.rstrip("\n"), add_special_tokens=False).input_ids


def _policy_tokens(tok, ref: dict, labelled_only: bool = False) -> int:
    """The policy tokens of a reference trajectory (calls turns + answer, what ``max_new_tokens`` counts)."""
    return (sum(len(tok(turn_text(t), add_special_tokens=False).input_ids) + 1 for t in ref["turns"]
                if t.get("label", True) or not labelled_only)
            + len(tok(ref["answer_reasoned"], add_special_tokens=False).input_ids) + 1)


def cmd_check_harness(args) -> None:
    """CPU checks of the rollout plumbing on the blind reference episodes of a split: the calls parse
    back to what was written, the pool reproduces the gold results, and the episode built by token
    concatenation (convert turn, simulate turn, compare turn, answer) is the chat template's own rendering.
    07: the same for the split's recovery episodes (read from ``reference --recover 1``), whose fix turn
    carries a line before its calls, and the forced probe's prefill reproduced through ``rollout``'s path."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    prefix = generation_prefix(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cases = load_split(args.split)[: args.n]
    pool = ToolPool(args.workers)
    same_tokens, roundtrip, totals, identical, n_turns, n_conv, n_comp = 0, 0, [], 0, Counter(), 0, 0
    try:
        for c in cases:
            ref = reference_episode(c, pool)
            n_turns[len(ref["turns"])] += 1
            n_conv += ref["converts"]
            n_comp += ref["compares"]
            by_key = {p["key"]: p for p in c["points"]}
            exact = pool.call_many([(c["tool"], by_key[k]["args"]) for k in c["proof"]])
            identical += sum(r == by_key[k]["result"] for r, k in zip(exact, c["proof"]))
            ok, same = _template_check(tok, c, ref, prefix, im_end)
            roundtrip += ok
            same_tokens += same
            totals.append(ref["reward"])
    finally:
        pool.close()
    n = len(cases)
    out = {"split": args.split, "n": n, "calls_parse_roundtrip": roundtrip, "token_ids_equal_template": same_tokens,
           "pool_results_identical_to_gold_exact_args": identical, "pool_calls": sum(len(c["proof"]) for c in cases),
           "episodes_by_call_turns": dict(n_turns), "convert_calls": n_conv, "compare_calls": n_comp,
           "reference_score_through_pool_min": min(totals), "reference_score_through_pool_mean": sum(totals) / n}
    # 07: the recovery episodes of the split, through the same checks.
    rpath = recovery_path(args.split)
    if rpath.exists():
        by_id = {c["id"]: c for c in load_split(args.split)}
        recs = [json.loads(line) for line in rpath.read_text().splitlines()][: args.n]
        checks = [_template_check(tok, by_id[r["id"]], r, prefix, im_end) for r in recs]
        out["recovery"] = {"n": len(recs), "calls_parse_roundtrip": sum(a for a, _ in checks),
                           "token_ids_equal_template": sum(b for _, b in checks),
                           "slipped_turns_all_errors": sum(all(result_is_error(x) for x in t["results"])
                                                           for r in recs for t in r["turns"][:2] if not t.get("label", True)
                                                           and t["calls"][0]["name"] in TOOLS),
                           "reward_min": min(r["reward"] for r in recs), "reward_mean": sum(r["reward"] for r in recs) / len(recs)}
    # 06: the policy's token budget must hold the reference trajectory itself (calls turns + answer, what
    # ``max_new_tokens`` counts). Measured on every train reference episode (the seed's and PPO's split);
    # 07: and on every train recovery episode, slipped turns included (the policy writes them when it slips).
    budget = PPO_DEFAULTS["max_new_tokens"]
    for name, path in (("train_reference_policy_tokens", reference_path("train")),
                       ("train_recovery_policy_tokens", recovery_path("train"))):
        if path.exists():
            lens = sorted(_policy_tokens(tok, json.loads(line)) for line in path.read_text().splitlines())
            out[name] = {"episodes": len(lens), "median": lens[len(lens) // 2], "p90": lens[int(0.9 * len(lens))],
                         "max": lens[-1], "budget": budget, "over_budget": sum(x > budget for x in lens)}
    out["compare_tool"] = COMPARE_ON
    (HERE / f"{PREFIX}_harness_checks{'' if COMPARE_ON else '_nocompare'}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def forced_prefill(split: str) -> dict[str, list[dict]]:
    """07: the slipped turns of the split's recovery episodes, keyed by question -- what ``probe --forced 1``
    starts each episode from."""
    return {r["id"]: [t for t in r["turns"] if not t.get("label", True)]
            for r in map(json.loads, recovery_path(split).read_text().splitlines())}


def cmd_probe(args) -> None:
    """Run a policy on a split with the simulators in the loop; write summary + episodes.
    07: ``--forced 1`` runs only the questions with a recovery episode, each from its slipped turns."""
    import torch
    cap_device_memory(args.max_vram_mib)
    torch.manual_seed(args.seed)
    if args.tools_json:  # another notebook's declarations, e.g. 05's to probe its seed under its own prompt
        global _DECLS
        _DECLS = json.loads(Path(args.tools_json).read_text())
    policy, tok = load_policy(args.adapter, merge=bool(args.merge) and args.generator == "torch")
    gen = None
    if args.generator == "mlx":  # 08: generated by mlx-lm, the adapter folded in (MLXPolicy); PyTorch's model let go
        import mlx.core as mx
        gen = MLXPolicy()
        if args.adapter:
            gen.sync(policy)
        del policy
        release_device_cache()
        mx.random.seed(args.seed)
    cases = load_split(args.split)
    prefill = None
    if args.forced:
        prefill = forced_prefill(args.split)
        cases = [c for c in cases if c["id"] in prefill]
    if args.swap:
        cases = [swap_sentences(c, args.swap) for c in cases]
    if args.limit:
        rng = random.Random(0)
        cases = rng.sample(cases, min(args.limit, len(cases)))
    cases = [c for c in cases for _ in range(args.samples)]
    out_dir = RUNS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.split}-T{args.temperature:g}" + (f"-x{args.samples}" if args.samples > 1 else "")
           + (f"-n{args.limit}" if args.limit else "") + ("-stop" if args.stop_at_json else "")
           + (f"-swap-{args.swap}" if args.swap else "") + ("-forced" if args.forced else ""))
    pool = ToolPool(args.workers)
    episodes, scores, t0 = [], [], time.time()
    try:
        with (out_dir / f"{tag}.episodes.jsonl").open("w") as fh:
            for b in range(0, len(cases), args.batch):
                if gen is not None:
                    eps = rollout_mlx(gen, tok, cases[b: b + args.batch], pool, temperature=args.temperature,
                                      max_new_tokens=args.max_new_tokens, stop_at_json=bool(args.stop_at_json),
                                      prefill=prefill)
                else:
                    eps = rollout(policy, tok, cases[b: b + args.batch], pool, temperature=args.temperature,
                                  max_new_tokens=args.max_new_tokens, stop_at_json=bool(args.stop_at_json),
                                  prefill=prefill)
                for ep in eps:
                    sc = score_episode(ep)
                    episodes.append(ep)
                    scores.append(sc)
                    fh.write(json.dumps(ep.to_json() | {"score": {k: sc[k] for k in ("total", "components", "evidence", "decision_correct")}}) + "\n")
                done = b + len(eps)
                rb = recovery_block([recovery_stats(e.calls) for e in episodes])
                print(f"{done}/{len(cases)}  reward so far {sum(s['total'] for s in scores) / done:.3f}  "
                      f"erred {rb['sim_erred']:.2f} recovered {rb['recovered_given_error'] if rb['recovered_given_error'] is None else round(rb['recovered_given_error'], 2)}  "
                      f"{time.time() - t0:.0f}s  solves {pool.solved}", flush=True)
    finally:
        pool.close()
    summary = summarise(episodes, scores)
    summary["meta"] = {"adapter": args.adapter, "split": args.split, "temperature": args.temperature,
                       "samples": args.samples, "max_new_tokens": args.max_new_tokens, "limit": args.limit,
                       "stop_at_json": bool(args.stop_at_json), "forced": bool(args.forced), "device": pick_device(),
                       "merged_adapter": bool(args.merge and args.adapter), "generator": args.generator,
                       "seconds": time.time() - t0, "solves": pool.solved, "peak_device_mib": peak_device_mib()}
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["overall"], indent=1))



def cmd_reference(args) -> None:
    """Blind reference episodes for a split (simulated through the pool), verified by the reward.
    07: ``--recover 1`` builds the split's recovery episodes instead (``recovery_episode``)."""
    pool = ToolPool(args.workers)
    t0 = time.time()
    try:
        if args.recover:
            recs, why = build_recoveries(args.split, pool)
        else:
            refs = build_references(args.split, pool)
    finally:
        pool.close()
    DATA.mkdir(exist_ok=True)
    if args.recover:
        path = recovery_path(args.split)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        slips = [r["slip"] for r in recs]
        slipped = Counter(f"{s['param']} [{s['unit']}]" for s in slips)
        print(f"{path.name}: {len(recs)} recovery episodes, {pool.solved} solves, {time.time() - t0:.0f}s; {dict(why)}")
        print(f"  reward min {min(r['reward'] for r in recs):.4f} mean {sum(r['reward'] for r in recs) / len(recs):.4f} "
              f"(the same questions' reference episodes: {sum(r['reference_reward'] for r in recs) / len(recs):.4f})")
        print(f"  slipped: {dict(slipped.most_common())}")
        print(f"  errors: {dict(Counter(s['error_kind'] for s in slips).most_common())}")
        return
    path = DATA / reference_path(args.split).name
    path.write_text("".join(json.dumps(r) + "\n" for r in refs))
    rewards = [r["reward"] for r in refs]
    conv = [r["converts"] for r in refs]
    comp = [r["compares"] for r in refs]
    print(f"{path.name}: {len(refs)} episodes, {pool.solved} solves, reward min {min(rewards):.4f} "
          f"mean {sum(rewards) / len(rewards):.4f}; convert calls mean {sum(conv) / len(conv):.2f} max {max(conv)}, "
          f"episodes without any {sum(c == 0 for c in conv)}; compare calls mean {sum(comp) / len(comp):.2f} max {max(comp)}")


def cmd_seed(args) -> None:
    cap_device_memory(args.max_vram_mib)
    train_seed(RUNS / args.out, label=args.label, reasoning=bool(args.reasoning), epochs=args.epochs, lr=args.lr,
               accum=args.accum, lora_r=args.lora_r, seed=args.seed, limit=args.limit,
               label_per_class=args.label_per_class, recover=args.recover, recovery_every=args.recovery_every,
               fix_lines=bool(args.fix_lines))



def cmd_ppo(args) -> None:
    cap_device_memory(args.max_vram_mib)
    cfg = dict(PPO_DEFAULTS)
    for k, v in vars(args).items():
        if k in cfg and v is not None:
            cfg[k] = type(PPO_DEFAULTS[k])(v) if PPO_DEFAULTS[k] is not None else v
    if cfg["value_init_bias"] < 0 or not cfg["start_from"]:
        raise SystemExit("pass --start-from and --value-init-bias (the seed's own sampled mean reward)")
    resolve_critic(cfg)
    train_ppo(cfg, RUNS / args.out, progress=lambda m: print(m, flush=True), resume=bool(args.resume))


def resolve_critic(cfg: dict) -> None:
    """08: a separate critic's layer and penalties left at 0 are critic-fit's choice for that model."""
    if cfg["critic"] == "policy":
        return
    if cfg["critic"] not in CRITICS:
        raise SystemExit(f"--critic must be policy or one of {', '.join(CRITICS)}, not {cfg['critic']!r}")
    if not (cfg["critic_layer"] and cfg["critic_ridge"] and cfg["critic_l2"]):
        if not CRITIC_FIT.exists():
            raise SystemExit("run critic-fit first, or pass --critic-layer, --critic-ridge and --critic-l2")
        best = json.loads(CRITIC_FIT.read_text())["candidates"][cfg["critic"]]["best"]
        cfg["critic_layer"] = cfg["critic_layer"] or best["layer"]
        cfg["critic_ridge"] = cfg["critic_ridge"] or best["ridge"]
        cfg["critic_l2"] = cfg["critic_l2"] or best["l2"]


# ---------------------------------------------------------------------------------------------
# 08: critic-fit -- which model, which layer, which prover.
# ---------------------------------------------------------------------------------------------

RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PROVER_KS = (1, 2, 3, 4, 6, 8, 16)


def critic_fit_episodes(args, tok) -> list[dict]:
    """08: critic-fit's episodes -- the PPO run's samples (train questions) and the held-out probes (dev questions),
    replayed (``replay``) and kept as text with each turn's start, each turn's return to go and the success. Cached in
    critic-fit/episodes.pt (a cache, not the record: git ignores it; delete it to rebuild)."""
    import torch
    out_dir = RUNS / "critic-fit"
    path = out_dir / "episodes.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    prefix = generation_prefix(tok)
    cases = {c["id"]: c for s in ("train", "dev") for c in load_split(s)}
    rows = [r | {"source": args.train_run, "split": "train"} for r in read_jsonl(HERE / args.train_run / "samples.jsonl")
            if not args.max_step or r["step"] < args.max_step]
    for f in args.heldout.split(","):
        rows += [r | {"source": f, "split": "dev", "step": None} for r in read_jsonl(HERE / f)]
    pool = ToolPool(args.workers)
    t0 = time.time()
    try:
        eps = replay(rows, cases, pool)
    finally:
        pool.close()
    data = []
    for r, ep in zip(rows, eps):
        sc = score_episode(ep)
        text, cuts = episode_text(ep, tok, prefix)
        R = turn_rewards(ep, sc)
        data.append({"source": r["source"], "split": r["split"], "step": r["step"], "id": r["id"], "family": r["family"],
                     "text": text, "cuts": cuts, "G": [sum(R[t:]) for t in range(len(R))], "y": int(success(sc)),
                     "reward": sc["total"], "reward_recorded": r["reward"] if "reward" in r else r["score"]["total"],
                     "errors_off": sum(c["recorded_error"] is not None
                                       and result_is_error(c["result"]) != bool(c["recorded_error"]) for c in ep.calls)})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"replayed {len(data)} episodes ({pool.solved} solves, {time.time() - t0:.0f}s): rewards off the recorded "
          f"{sum(abs(d['reward'] - d['reward_recorded']) > 1e-9 for d in data)}, call errors off the recorded "
          f"{sum(d['errors_off'] for d in data)}", flush=True)
    return data


def critic_fit_features(name: str, data: list[dict], args) -> dict:
    """08: one candidate's features of every state of critic-fit's episodes, at each of its CRITIC_FIT_LAYERS (rows in
    episode order, a row per turn). Cached in critic-fit/features-<name>.pt."""
    import gc
    import torch
    path = RUNS / "critic-fit" / f"features-{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    crit = SeparateCritic(name, CRITIC_FIT_LAYERS[name], pick_device(),
                          adapter=HERE / args.policy_adapter if name == "policy" else None)
    feats: dict[int, list] = {L: [] for L in crit.layers}
    t0 = time.time()
    for i, d in enumerate(data):
        got = crit.features(d["text"], d["cuts"])
        for L in crit.layers:
            feats[L].append(got[L])
        if (i + 1) % 50 == 0:
            release_device_cache()
        if (i + 1) % 100 == 0 or i + 1 == len(data):
            print(f"  {name}: {i + 1}/{len(data)} episodes, {crit.seconds / (i + 1):.2f}s each, "
                  f"{crit.tokens / max(crit.seconds, 1e-9):.0f} tokens/s, {time.time() - t0:.0f}s", flush=True)
    res = {"name": name, "repo": CRITICS.get(name, MODEL), "layers": list(crit.layers), "depth": crit.depth,
           "hidden_size": crit.hidden_size, "feats": {L: torch.cat(v) for L, v in feats.items()},
           "seconds_per_episode": crit.seconds / len(data), "tokens_per_second": crit.tokens / crit.seconds,
           "device_mib": peak_device_mib()}
    torch.save(res, path)
    del crit
    gc.collect()
    release_device_cache()
    return res


def _standardized(fit_rows, *others):
    mu, sd = fit_rows.mean(0), fit_rows.std(0).clamp(min=1e-6)
    return [(x - mu) / sd for x in (fit_rows, *others)]


def _logloss(p, y) -> float:
    p = p.clamp(1e-6, 1 - 1e-6)
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def _ev(pred, target) -> float | None:
    var = float(target.var(unbiased=False))
    return None if var < 1e-9 else 1.0 - float((target - pred).var(unbiased=False)) / var


def _by_turn(values, turns, fit_idx, idx, cap: int = 6):
    """The fitting rows' mean of ``values`` at each turn index (turns past ``cap`` pooled): a clock baseline."""
    import torch
    t_fit = [min(int(turns[k]), cap) for k in fit_idx]
    means = {}
    for t in set(t_fit):
        sel = [k for k, tk in zip(fit_idx, t_fit) if tk == t]
        means[t] = float(values[sel].mean())
    overall = float(values[fit_idx].mean())
    return torch.tensor([means.get(min(int(turns[k]), cap), overall) for k in idx], dtype=values.dtype)


def prover_table(q, y, turns, ep, adv) -> dict:
    """08: the paper's prover criterion (its Theorem 3.1: improvement >~ distinguishability + alignment) for BoK(pi)
    over K, from predicted success probabilities. Per K:
      D, distinguishability: the variance of A^mu over the actions at a state. One action per state is recorded, so
        its deviation from the state's mean is taken to first order, f_K'(q_t) (q_{t+1} - q_t) -- q is a martingale
        under pi, E[q_{t+1} | s_t] = q_t -- and exactly y - q at the last turn (f_K(1) - f_K(0) = 1 for every K). The
        plain variance over all turns would add the spread *between* states of A^mu's mean (-f_K(q) at a failing
        last turn, ~-1 for a large K), which is not a difference between actions and grows without end in K;
      C, alignment: the mean of A^mu A^pi, A^pi the base policy's own advantage in success (A^mu at K = 1; its mean at
        a state is 0, so the plain product estimates the covariance);
    their sum, the share of states BoK(pi) already calls won (f_K(q) > 0.95: a prover too strong to tell steps
    apart), and A^mu against PPO's advantage (``adv``: return to go less v) -- their correlation, and the scale alpha
    matching their spreads. ``q``, ``turns``, ``ep``, ``adv``: one entry per state; ``y``: {episode: success}."""
    import torch
    by_ep = defaultdict(list)
    for k in range(len(q)):
        by_ep[int(ep[k])].append((int(turns[k]), k))
    order, q_eps = [], []
    for e, items in by_ep.items():
        items.sort()
        order += [k for _, k in items]
        q_eps.append((torch.tensor([float(q[k]) for _, k in items], dtype=torch.float64), float(y[e])))
    a_ppo = adv[order].double()
    a_pi = torch.cat([prover_progress(qe, ye, 1) for qe, ye in q_eps])
    q_all = torch.cat([qe for qe, _ in q_eps])
    out = {}
    for k in PROVER_KS:
        a_mu = torch.cat([prover_progress(qe, ye, k) for qe, ye in q_eps])
        spread = torch.cat([torch.cat([k * (1 - qe[:-1]) ** (k - 1) * (qe[1:] - qe[:-1]),
                                       torch.tensor([ye], dtype=qe.dtype) - qe[-1:]]) for qe, ye in q_eps])
        D, C = float((spread ** 2).mean()), float((a_mu * a_pi).mean())
        sd = float(a_mu.std(unbiased=False))
        corr = float(torch.corrcoef(torch.stack([a_mu, a_ppo]))[0, 1]) if sd > 1e-9 else None
        out[k] = {"D": round(D, 6), "C": round(C, 6), "D_plus_C": round(D + C, 6), "mean": round(float(a_mu.mean()), 6),
                  "sd": round(sd, 6), "saturated": round(float((bok_value(q_all, k) > 0.95).double().mean()), 4),
                  "corr_ppo_adv": None if corr is None else round(corr, 4),
                  "alpha_matching_sd": None if sd < 1e-9 else round(float(a_ppo.std(unbiased=False)) / sd, 3)}
    return out


def critic_fit_layer(X, meta: dict, args) -> dict:
    """08: one candidate layer. Penalties by cross-validation on the train questions (grouped by question), then two
    tests: the held-out dev questions, fitted on every train episode; and the online replay -- for every `stride`-th
    step of the PPO run, the heads fitted on the `window` steps before it and tested on its episodes, as ``train_ppo``
    fits and uses them."""
    import zlib
    import torch
    X = X.double()
    G, y, turns = meta["G"], meta["y"], meta["t"]
    tr = [k for k, s in enumerate(meta["split"]) if s == "train"]
    dev = [k for k, s in enumerate(meta["split"]) if s == "dev"]
    fold = {k: zlib.crc32(meta["qid"][k].encode()) % args.folds for k in tr}
    splits = []
    for f in range(args.folds):
        a = [k for k in tr if fold[k] != f]
        b = [k for k in tr if fold[k] == f]
        Za, Zb = _standardized(X[a], X[b])
        splits.append((a, b, Za, Zb))
    cv_v, cv_q = {}, {}
    for lam in RIDGE_GRID:
        err = sum(float(((Zb @ w + b0 - G[b]) ** 2).sum()) for a, b, Za, Zb in splits
                  for w, b0 in [fit_ridge(Za, G[a], lam)])
        cv_v[lam] = err / len(tr)
    for lam in L2_GRID:
        tot = 0.0
        for a, b, Za, Zb in splits:
            w, b0 = fit_logistic(Za, y[a], lam)
            tot += _logloss(torch.sigmoid(Zb @ w + b0), y[b]) * len(b)
        cv_q[lam] = tot / len(tr)
    del splits
    ridge, l2 = min(cv_v, key=cv_v.get), min(cv_q, key=cv_q.get)
    res = {"ridge": ridge, "l2": l2, "cv_mse": {str(k): round(v, 6) for k, v in cv_v.items()},
           "cv_logloss": {str(k): round(v, 6) for k, v in cv_q.items()}}

    def test(fit_idx, idx) -> dict:
        Za, Zb = _standardized(X[fit_idx], X[idx])
        wv, bv = fit_ridge(Za, G[fit_idx], ridge)
        wq, bq = fit_logistic(Za, y[fit_idx], l2)
        return {"v": Zb @ wv + bv, "q": torch.sigmoid(Zb @ wq + bq),
                "v_clock": _by_turn(G, turns, fit_idx, idx), "q_clock": _by_turn(y, turns, fit_idx, idx),
                "base": float(y[fit_idx].mean())}

    def report(idx, got) -> dict:
        yy, GG, q = y[idx], G[idx], got["q"]
        base = got["base"] if torch.is_tensor(got["base"]) else torch.full_like(yy, got["base"])
        first = [n for n, k in enumerate(idx) if int(turns[k]) == 0]
        later = [n for n, k in enumerate(idx) if int(turns[k]) > 0]
        return {"states": len(idx), "success": round(float(yy.mean()), 4),
                "q_logloss": round(_logloss(q, yy), 5), "q_logloss_clock": round(_logloss(got["q_clock"], yy), 5),
                "q_logloss_base": round(_logloss(base, yy), 5),
                "q_brier": round(float(((q - yy) ** 2).mean()), 5),
                "q_auc": auc(q.tolist(), yy.int().tolist()),
                "q_auc_first_turn": auc(q[first].tolist(), yy[first].int().tolist()),
                "q_auc_later_turns": auc(q[later].tolist(), yy[later].int().tolist()),
                "v_ev": _ev(got["v"], GG), "v_ev_clock": _ev(got["v_clock"], GG)}

    got = test(tr, dev)
    res["heldout"] = report(dev, got)
    pred = {"heldout": (dev, got)}
    steps = sorted({int(meta["step"][k]) for k in tr})
    idx_all, parts = [], defaultdict(list)
    for s in steps:
        if s < args.window or (s - args.window) % args.stride:
            continue
        fit_idx = [k for k in tr if s - args.window <= meta["step"][k] < s]
        idx = [k for k in tr if meta["step"][k] == s]
        if not idx:
            continue
        g = test(fit_idx, idx)
        idx_all += idx
        for key in ("v", "q", "v_clock", "q_clock"):
            parts[key].append(g[key])
        parts["base"].append(torch.full((len(idx),), g["base"], dtype=torch.float64))  # each step's own window rate
    online = {key: torch.cat(v) for key, v in parts.items()}
    res["online"] = report(idx_all, online) if idx_all else None
    pred["online"] = (idx_all, online)
    for name, (idx, got_) in pred.items():
        if idx:
            res.setdefault("prover", {})[name] = prover_table(
                got_["q"], {int(meta["ep"][k]): float(y[k]) for k in idx}, [turns[k] for k in idx],
                [meta["ep"][k] for k in idx], G[idx] - got_["v"])
    return res


def cmd_critic_fit(args) -> None:
    """08: the candidate critics compared on 07's sampled PPO episodes (online replay, and held-out dev questions), and
    the paper's prover criterion over K; writes 08_prover_critic_critic_fit.json."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)  # the policy's chat template writes every episode's text
    data = critic_fit_episodes(args, tok)
    meta = {"ep": [], "t": [], "split": [], "step": [], "qid": [], "G": [], "y": []}
    for i, d in enumerate(data):
        for t in range(len(d["cuts"])):
            meta["ep"].append(i)
            meta["t"].append(t)
            meta["split"].append(d["split"])
            meta["step"].append(d["step"])
            meta["qid"].append(d["id"])
            meta["G"].append(d["G"][t])
            meta["y"].append(float(d["y"]))
    meta["G"], meta["y"] = (torch.tensor(meta[k], dtype=torch.float64) for k in ("G", "y"))
    train = [d for d in data if d["split"] == "train"]
    out = {"data": {"train_run": args.train_run, "heldout": args.heldout.split(","), "episodes": len(data),
                    "train_episodes": len(train), "heldout_episodes": len(data) - len(train),
                    "train_steps": [min(d["step"] for d in train), max(d["step"] for d in train)],
                    "states": len(meta["ep"]), "success_train": sum(d["y"] for d in train) / len(train),
                    "success_heldout": sum(d["y"] for d in data if d["split"] == "dev") / max(1, len(data) - len(train)),
                    "rewards_off_recorded": sum(abs(d["reward"] - d["reward_recorded"]) > 1e-9 for d in data),
                    "call_errors_off_recorded": sum(d["errors_off"] for d in data),
                    "window": args.window, "stride": args.stride, "folds": args.folds},
           "candidates": {}}
    if CRITIC_FIT.exists():  # candidates fitted by an earlier call stay unless refitted
        out["candidates"] = json.loads(CRITIC_FIT.read_text()).get("candidates", {})
    for name in args.candidates.split(","):
        if args.cached_only and not (RUNS / "critic-fit" / f"features-{name}.pt").exists():
            continue
        f = critic_fit_features(name, data, args)
        if args.features_only:
            continue
        cand = {k: f[k] for k in ("repo", "depth", "hidden_size", "seconds_per_episode", "tokens_per_second",
                                   "device_mib")} | {"layers": {}}
        for L in f["layers"]:
            t0 = time.time()
            cand["layers"][str(L)] = r = critic_fit_layer(f["feats"][L], meta, args)
            on, ho = r["online"], r["heldout"]
            s = lambda x, spec: "-" if x is None else format(x, spec)  # noqa: E731
            print(f"{name} layer {L}: online q logloss {on['q_logloss']:.4f} (base {on['q_logloss_base']:.4f}, clock "
                  f"{on['q_logloss_clock']:.4f}) auc {s(on['q_auc'], '.3f')} v ev {s(on['v_ev'], '+.3f')} (clock "
                  f"{s(on['v_ev_clock'], '+.3f')}) | heldout q logloss {ho['q_logloss']:.4f} auc {s(ho['q_auc'], '.3f')} "
                  f"v ev {s(ho['v_ev'], '+.3f')} | ridge {r['ridge']:g} l2 {r['l2']:g} | {time.time() - t0:.0f}s",
                  flush=True)
        best = min(cand["layers"], key=lambda L: cand["layers"][L]["online"]["q_logloss"])
        cand["best"] = {"layer": int(best), "ridge": cand["layers"][best]["ridge"], "l2": cand["layers"][best]["l2"]}
        out["candidates"][name] = cand
        CRITIC_FIT.write_text(json.dumps(out, indent=1) + "\n")
    # The choice, by a rule fixed before the numbers: the separate candidate whose best layer has the lowest online
    # q log-loss -- the prover's input, measured as PPO will use it -- the cheaper one when within `tolerance` nats;
    # K the maximiser of D + C on the same online predictions; alpha the scale matching A^mu's spread to PPO's
    # advantage's (the paper tuned alpha on validation runs, 0.5-6 all helping; this machine affords none).
    # Added after the first fit, when the maximiser came out at the grid's edge (K = 16): an edge is no turning point.
    # There D was still rising only because it is estimated from q's own differences -- the critic's noise at low q,
    # amplified by f_K'(q)^2 ~ K^2 -- while C peaked at K = 2-3, A^mu's correlation with PPO's advantage fell with every
    # K, and BoK(pi) already called half the states won (the paper's too-strong prover, its A^mu ~ 0 but at a failing
    # last turn). At the edge the paper's own Bo4 is taken.
    choose_critic(out, args.tolerance)
    CRITIC_FIT.write_text(json.dumps(out, indent=1) + "\n")


def choose_critic(out: dict, tolerance: float) -> None:
    """08: critic-fit's choice (``cmd_critic_fit`` gives the rule), written into ``out["choice"]``."""
    sep = {n: c for n, c in out["candidates"].items() if n != "policy" and "best" in c}
    if not sep:
        return
    ll = {n: c["layers"][str(c["best"]["layer"])]["online"]["q_logloss"] for n, c in sep.items()}
    ok = [n for n in sep if ll[n] <= min(ll.values()) + tolerance]
    name = min(ok, key=lambda n: sep[n]["seconds_per_episode"])
    layer = sep[name]["best"]["layer"]
    table = {int(k): v for k, v in sep[name]["layers"][str(layer)]["prover"]["online"].items()}
    k_crit = max(table, key=lambda kk: table[kk]["D_plus_C"])
    k = 4 if k_crit == max(PROVER_KS) else k_crit
    out["choice"] = {"critic": name, "layer": layer, "ridge": sep[name]["best"]["ridge"], "l2": sep[name]["best"]["l2"],
                     "prover_k": k, "prover_k_criterion": k_crit, "k_at_grid_edge": k_crit == max(PROVER_KS),
                     "alpha": round(table[k]["alpha_matching_sd"], 1), "tolerance": tolerance, "online_q_logloss": ll}
    print(f"choice: {out['choice']}")


def cmd_device_bench(args) -> None:
    """07: what a PPO step costs on this machine, piece by piece, and whether the MPS delta rule computes what
    transformers' does. On one train reference episode: the forward with log-probs at its labelled tokens and
    the checkpointed forward+backward (the actor's pass), with transformers' rule and 07's; the log-probs of
    both against the model in float32 (bf16's own noise); and generation, 8 real prompts, 1 and 256 new
    tokens, sampled. Writes 07_recover_from_errors_device.json."""
    import platform
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device()
    original = qwen35.torch_chunk_gated_delta_rule

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def timed(fn):
        fn()
        sync()
        t = time.time()
        out = fn()
        sync()
        return out, round(time.time() - t, 2)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    cases = {c["id"]: c for c in load_split("train")}
    lines = reference_path("train").read_text().splitlines()
    ref = json.loads(lines[args.episode])
    ids, labels = (x.to(device) for x in seed_example(cases[ref["id"]], ref, tok, label="full", reasoning=True))
    where = labels[:, 1:] != -100

    def logprobs():
        with torch.no_grad():
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
            lp = torch.log_softmax(model.lm_head(h[where]).float(), -1)
            return lp.gather(-1, labels[:, 1:][where].unsqueeze(-1)).squeeze(-1)

    def fwd_bwd():
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
        loss = torch.nn.functional.cross_entropy(model.lm_head(h[where]).float(), labels[:, 1:][where])
        loss.backward()
        return float(loss.detach())

    rules = [("transformers", original)] + ([("07", mps_chunk_gated_delta_rule)] if device == "mps" else [])
    out = {"machine": platform.machine(), "system": platform.system(), "processor": platform.processor(),
           "device": device, "torch": torch.__version__, "episode": ref["id"], "episode_tokens": int(ids.shape[1]),
           "labelled_tokens": int(where.sum()), "forward_s": {}, "forward_backward_checkpointed_s": {}}
    lps = {}
    for name, fn in rules:
        qwen35.torch_chunk_gated_delta_rule = fn
        lps[name], out["forward_s"][name] = timed(logprobs)
    model.float()
    qwen35.torch_chunk_gated_delta_rule = original
    lp32, _ = timed(logprobs)
    model.to(torch.bfloat16)
    out["logprob_vs_float32"] = {name: {"max": round(float((lp - lp32).abs().max()), 4),
                                        "mean": round(float((lp - lp32).abs().mean()), 5)} for name, lp in lps.items()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    for name, fn in rules:
        qwen35.torch_chunk_gated_delta_rule = fn
        out["forward_backward_checkpointed_s"][name] = timed(fwd_bwd)[1]
    model.gradient_checkpointing_disable()
    model.eval()
    qwen35.torch_chunk_gated_delta_rule = rules[-1][1]
    batch = [cases[json.loads(line)["id"]] for line in lines[:8]]
    prompts = [tok(render_prompt(tok, c), add_special_tokens=False).input_ids for c in batch]
    width, pad = max(map(len, prompts)), tok.pad_token_id
    inp = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
    att = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    out["generate_8_prompts"] = {"prompt_tokens": width, "rule": rules[-1][0]}
    for new in (1, 256):
        def gen():
            with torch.no_grad():
                return model.generate(input_ids=inp, attention_mask=att, do_sample=True, temperature=1.0, top_k=0,
                                      top_p=1.0, max_new_tokens=new, min_new_tokens=new, pad_token_id=pad)
        out["generate_8_prompts"][f"{new}_new_s"] = timed(gen)[1]
    qwen35.torch_chunk_gated_delta_rule = original
    (HERE / f"{PREFIX}_device.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def cmd_worker(_args) -> None:
    host = _host()
    for line in sys.stdin:
        req = json.loads(line)
        sys.stdout.write(json.dumps({"id": req.get("id"), "result": host.call(req["name"], req.get("arguments"))}) + "\n")
        sys.stdout.flush()


def cmd_xcheck(args) -> None:
    """Re-solve a sample of gold points through the deployed MCP (respects its 30/min limit)."""
    import asyncio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    env = dict(line.split("=", 1) for line in (REPO / ".env").read_text().splitlines()
               if "=" in line and not line.startswith("#"))
    url, token = env["RO_MCP_URL"].strip(), env["MCP_BEARER_TOKEN"].strip()
    cases = load_split(args.split)
    rng = random.Random(0)
    sample = [(c, rng.choice(c["points"])) for c in rng.sample(cases, args.n)]

    async def run():
        rows = []
        async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                for c, p in sample:
                    res = await s.call_tool(c["tool"], p["args"])
                    rows.append((c["id"], p["key"], res.content[0].text == p["result"]))
                    await asyncio.sleep(2.5)
        return rows

    rows = asyncio.run(run())
    for row in rows:
        print(*row)
    print(f"identical result text: {sum(r[2] for r in rows)}/{len(rows)}")
    out = {"endpoint": url, "split": args.split, "identical": sum(r[2] for r in rows), "n": len(rows),
           "rows": [{"id": i, "point": k, "identical": same} for i, k, same in rows]}
    (HERE / f"{PREFIX}_xcheck.json").write_text(json.dumps(out, indent=1) + "\n")


# ----------------------------------------------------------------------------------------------
# The probe on Apple silicon (mlx-lm), for models this notebook does not train. Prompt, tool pool,
# episode bookkeeping, scoring and summary are the ones above; only the generation step differs:
# one episode at a time, greedy.
#   * The model's state over the tokens every prompt starts with (system message + tool declarations)
#     is computed once and copied per episode; each turn then feeds only the tokens that are new.
#   * Activations run in float32 (``--f32 1``), the weights as stored. In bf16 the logits of
#     Qwen3.5-4B move by about 0.5 with how the same context is cut into pieces, and greedy episodes
#     with and without the cache part ways; in float32 they are token-identical.
#   * ``--stop-at-json 1`` is ``probe``'s rule: a call-free turn ends where its first top-level {...}
#     closes. An untrained model also writes braces that are not an answer (LaTeX: 10^{-12}), and the
#     rule ends those turns in the middle of a sentence. ``--stop-at-json 2`` ends the turn only where
#     an answer object closes. With ``--reuse <state of a rule-1 run>`` only the episodes rule 1 ended
#     on a brace that was not an answer are generated again -- greedy decoding makes every other
#     episode the same under both rules -- and ``--verify N`` regenerates N of the reused ones and
#     compares their tokens.
#   * A run that is interrupted continues from its .state.jsonl when started again with the same
#     arguments.
# ----------------------------------------------------------------------------------------------
#: Where the seconds go: reading context, writing tokens, copying the shared prefix, waiting for tools.
MLX_CLOCK = {"turns": 0, "prefill_tokens": 0, "prefill_s": 0.0, "new_tokens": 0, "generate_s": 0.0, "copy_s": 0.0,
             "tool_s": 0.0}


def mlx_answer_closed(text: str, rule: int) -> bool:
    """Has this turn's answer closed? rule 1: its first top-level {...}; rule 2: an answer object."""
    if "<tool_call>" in text:
        return False
    return bool(json_spans(text)) if rule == 1 else extract_answer(text)[0] is not None


def mlx_turn(model, hf_tok, ids: list[int], budget: int, *, im_end: int, eot: int, stop_at_json: int,
             cache: dict, prompt_cache=None, temperature: float = 0.0) -> tuple[list[int], bool, bool]:
    """One policy turn: (tokens taken, ended by <|im_end|>, answer closed at its JSON). Greedy, or sampled at
    ``temperature`` (08: PPO's rollouts, the whole distribution as ``rollout``'s generate: no top-k / top-p) with
    <|endoftext|> suppressed as ``rollout`` suppresses it (its log-prob is -1e9 in ``forward_policy``).
    With ``prompt_cache``, ``ids`` is only what the cache has not seen; mlx-lm feeds every token it
    yields into the cache before yielding it, so afterwards the cache holds ids + the tokens taken."""
    import os
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    ended = closed = False
    out, text, t0, last = [], "", time.time(), None
    kw = {} if prompt_cache is None else {"prompt_cache": prompt_cache}
    if temperature > 0:
        def no_eot(_tokens, logits):
            return logits + mx.where(mx.arange(logits.shape[-1]) == eot, -mx.inf, 0.0).astype(logits.dtype)
        kw["logits_processors"] = [no_eot]
    for r in stream_generate(model, hf_tok, prompt=mx.array(ids), max_tokens=budget,
                             sampler=make_sampler(temp=temperature), **kw):
        t, last = r.token, r
        if t == eot:  # ``probe`` suppresses <|endoftext|>; here it ends the turn
            break
        out.append(t)
        if t == im_end:
            ended = True
            break
        if stop_at_json:
            piece = _tok_text(hf_tok, t, cache)
            text += piece
            if "}" in piece and mlx_answer_closed(text, stop_at_json):
                closed = True
                break
    prefill = last.prompt_tokens / last.prompt_tps if last is not None and last.prompt_tps else 0.0
    if os.environ.get("MLX_PROBE_DEBUG"):
        print(f"  turn: read {len(ids)} tok in {prefill:.1f}s, wrote {len(out)} tok in {time.time() - t0 - prefill:.1f}s "
              f"({last.generation_tps:.1f} tok/s by mlx-lm), peak {last.peak_memory:.1f} GB", flush=True)
    MLX_CLOCK["turns"] += 1
    MLX_CLOCK["prefill_tokens"] += len(ids)
    MLX_CLOCK["prefill_s"] += prefill
    MLX_CLOCK["new_tokens"] += len(out)
    MLX_CLOCK["generate_s"] += time.time() - t0 - prefill
    return out, ended, closed


def mlx_shared_prefix(model, prompts: list[list[int]]):
    """The model's state after the tokens all prompts share (at least one token is left to each)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    n = min(len(p) for p in prompts) - 1
    for p in prompts[1:]:
        n = next((i for i in range(n) if p[i] != prompts[0][i]), n)
    base = make_prompt_cache(model)
    for _ in generate_step(mx.array(prompts[0][:n]), model, max_tokens=0, prompt_cache=base):
        pass
    mx.eval([c.state for c in base])
    return base, prompts[0][:n]


def mlx_rollout(model, hf_tok, case: dict, pool, *, max_new_tokens: int, stop_at_json: int, cache: dict,
                prefix: str, im_end: int, eot: int, base=None):
    """One episode, ``rollout``'s rules for one row: (prompt, turn, tool results, turn, ...)."""
    import copy
    ep = Episode(case, hf_tok(render_prompt(hf_tok, case), add_special_tokens=False).input_ids)
    prompt_cache, fed = None, 0  # fed = how many tokens of prompt_ids + ids the cache already holds
    if base is not None:
        assert ep.prompt_ids[:len(base[1])] == base[1]
        t0 = time.time()
        prompt_cache, fed = copy.deepcopy(base[0]), len(base[1])
        MLX_CLOCK["copy_s"] += time.time() - t0
    for rnd in range(MAX_ROUNDS + 1):
        budget = max_new_tokens - ep.policy_tokens
        if budget <= 0:
            ep.truncated = True
            break
        context = ep.prompt_ids + ep.ids
        take, ended, closed = mlx_turn(model, hf_tok, context[fed:], budget, im_end=im_end, eot=eot,
                                       stop_at_json=stop_at_json, cache=cache, prompt_cache=prompt_cache)
        if prompt_cache is not None:
            fed = len(context) + len(take)
        ep.turn_starts.append(len(ep.ids))
        ep.ids += take
        ep.policy += [True] * len(take)
        text = hf_tok.decode(take, skip_special_tokens=True)
        ep.turns.append(text)
        if closed:
            ep.final_text, ep.answer_closed = text, True
            break
        if not ended:
            ep.truncated, ep.final_text = True, text
            break
        calls = parse_calls(text)
        if not calls:
            ep.final_text = text
            break
        if rnd == MAX_ROUNDS:
            ep.unanswered = len(calls)
            break
        n_sims = sum(x.get("name") not in LOCAL_TOOLS for x in ep.calls)
        n_conv = sum(x.get("name") == CONVERT for x in ep.calls)
        n_comp = sum(x.get("name") == COMPARE for x in ep.calls)
        jobs, where = [], []
        for c in calls:
            if c.get("malformed"):
                c["result"] = MALFORMED_CALL_ERROR
            elif c.get("name") == CONVERT:
                n_conv += 1
                if n_conv > MAX_CONVERTS:
                    c["result"] = CONVERT_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"]))
                    where.append(c)
            elif c.get("name") == COMPARE:
                n_comp += 1
                if n_comp > MAX_COMPARES:
                    c["result"] = COMPARE_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"]))
                    where.append(c)
            else:
                n_sims += 1
                if n_sims > MAX_CALLS:
                    c["result"] = CALL_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"]))
                    where.append(c)
        t0 = time.time()
        for c, result in zip(where, pool.call_many(jobs)):
            c["result"] = result
        MLX_CLOCK["tool_s"] += time.time() - t0
        for c in calls:
            c["turn"] = len(ep.turns) - 1
            ep.calls.append(c)
        env = hf_tok(tool_response_text([c["result"] for c in calls], prefix), add_special_tokens=False).input_ids
        ep.ids += env
        ep.policy += [False] * len(env)
    return ep


class MLXPolicy:
    """08: the policy's generation on MLX (PPO's rollouts and evaluations; training stays in PyTorch). The base model
    loaded once by mlx-lm; ``sync`` folds the PyTorch policy's current LoRA into it -- W = W0 + (alpha / r) B A for
    every adapted linear, the same module names in both -- before each rollout. Log-probabilities stay PyTorch's
    (``forward_policy`` reads every sampled episode again), so the update is PPO's either way; MLX only samples."""

    def __init__(self):
        import mlx.nn as nn
        from mlx_lm import load
        self.model, self.tok = load(MODEL)
        self.linears = {n: m for n, m in self.model.named_modules() if isinstance(m, nn.Linear)}
        self.base: dict = {}  # each adapted linear's pretrained weight, kept to fold the next step's LoRA into
        self.seconds = 0.0

    def _mlx_name(self, torch_name: str) -> str:
        # PEFT: base_model.model.model.layers.N.<...>; mlx-lm's Qwen3.5: language_model.model.layers.N.<...>
        return "language_model.model." + torch_name[torch_name.index("layers."):]

    def sync(self, policy) -> int:
        import mlx.core as mx
        import torch
        t0 = time.time()
        n = 0
        with torch.no_grad():
            for name, module in policy.named_modules():
                if not (hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {})):
                    continue
                key = self._mlx_name(name)
                lin = self.linears[key]
                if key not in self.base:
                    self.base[key] = lin.weight
                A = module.lora_A["default"].weight.float()
                B = module.lora_B["default"].weight.float()
                delta = (B @ A) * module.scaling["default"]
                d = mx.array(delta.cpu().numpy())
                lin.weight = (self.base[key].astype(mx.float32) + d).astype(self.base[key].dtype)
                n += 1
        mx.eval([self.linears[k].weight for k in self.base])
        self.seconds += time.time() - t0
        return n


def rollout_mlx(gen: MLXPolicy, tokenizer, cases: list[dict], pool: ToolPool, *, temperature: float,
                max_new_tokens: int, stop_at_json: bool = False,
                prefill: dict[str, list[dict]] | None = None) -> list[Episode]:
    """08: ``rollout`` with the generation on MLX, episode by episode within each round, every episode keeping its own
    prompt cache across its turns (the system turn read once a call and copied); the rounds, the budgets, the tool
    calls (each round's, every episode's, on the pool at once) and the Episode records are ``rollout``'s, ``prefill``
    (the forced probe's slipped turns) included."""
    import copy
    import mlx.core as mx
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    prefix = generation_prefix(tokenizer)
    cache: dict[int, str] = {}
    eps = [Episode(c, tokenizer(render_prompt(tokenizer, c), add_special_tokens=False).input_ids) for c in cases]
    used = [0] * len(eps)  # tool rounds already taken by prefilled turns (``rollout``'s)
    for i, ep in enumerate(eps):
        for turn in (prefill or {}).get(ep.case["id"], []):
            text = turn_text(turn)
            ep.turn_starts.append(len(ep.ids))
            ep.turns.append(text)
            for c, r in zip(turn["calls"], turn["results"]):
                ep.calls.append({"name": c["name"], "arguments": c["arguments"], "result": r, "turn": len(ep.turns) - 1})
            written = tokenizer(text, add_special_tokens=False).input_ids + [im_end]
            env = tokenizer(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
            ep.ids += written + env
            ep.policy += [False] * (len(written) + len(env))
            used[i] += 1
    base = mlx_shared_prefix(gen.model, [ep.prompt_ids for ep in eps])
    caches = [copy.deepcopy(base[0]) for _ in eps]
    fed = [len(base[1])] * len(eps)
    live = list(range(len(eps)))
    for rnd in range(MAX_ROUNDS + 1):
        budgets = {i: max_new_tokens - eps[i].policy_tokens for i in live}
        for i in live:
            if budgets[i] <= 0:
                eps[i].truncated = True
        live = [i for i in live if budgets[i] > 0]
        if not live:
            break
        pending = []
        for i in live:
            ep = eps[i]
            context = ep.prompt_ids + ep.ids
            take, ended, closed = mlx_turn(gen.model, gen.tok, context[fed[i]:], budgets[i], im_end=im_end, eot=eot,
                                           stop_at_json=int(stop_at_json), cache=cache, prompt_cache=caches[i],
                                           temperature=temperature)
            fed[i] = len(context) + len(take)
            ep.turn_starts.append(len(ep.ids))
            ep.ids += take
            ep.policy += [True] * len(take)
            text = tokenizer.decode(take, skip_special_tokens=True)
            ep.turns.append(text)
            if closed:
                ep.final_text, ep.answer_closed = text, True
                continue
            if not ended:
                ep.truncated, ep.final_text = True, text
                continue
            calls = parse_calls(text)
            if not calls:
                ep.final_text = text
                continue
            if rnd + used[i] >= MAX_ROUNDS:
                ep.unanswered = len(calls)
                continue
            pending.append((i, calls))
        jobs, where = [], []
        for i, calls in pending:
            n_sims = sum(x.get("name") not in LOCAL_TOOLS for x in eps[i].calls)
            n_conv = sum(x.get("name") == CONVERT for x in eps[i].calls)
            n_comp = sum(x.get("name") == COMPARE for x in eps[i].calls)
            for c in calls:
                if c.get("malformed"):
                    c["result"] = MALFORMED_CALL_ERROR
                elif c.get("name") == CONVERT:
                    n_conv += 1
                    if n_conv > MAX_CONVERTS:
                        c["result"] = CONVERT_BUDGET_ERROR
                    else:
                        jobs.append((c["name"], c["arguments"]))
                        where.append(c)
                elif c.get("name") == COMPARE:
                    n_comp += 1
                    if n_comp > MAX_COMPARES:
                        c["result"] = COMPARE_BUDGET_ERROR
                    else:
                        jobs.append((c["name"], c["arguments"]))
                        where.append(c)
                else:
                    n_sims += 1
                    if n_sims > MAX_CALLS:
                        c["result"] = CALL_BUDGET_ERROR
                    else:
                        jobs.append((c["name"], c["arguments"]))
                        where.append(c)
        for c, result in zip(where, pool.call_many(jobs)):
            c["result"] = result
        live = []
        for i, calls in pending:
            ep = eps[i]
            for c in calls:
                c["turn"] = len(ep.turns) - 1
                ep.calls.append(c)
            env = tokenizer(tool_response_text([c["result"] for c in calls], prefix), add_special_tokens=False).input_ids
            ep.ids += env
            ep.policy += [False] * len(env)
            live.append(i)
    del caches, base
    mx.clear_cache()
    return eps


def mlx_episode_row(ep, sc: dict) -> dict:
    """The probe's row, with each call's result text beside its error flag."""
    row = ep.to_json() | {"score": {k: sc[k] for k in ("total", "components", "evidence", "decision_correct")}}
    for c, full in zip(row["calls"], ep.calls):
        c["result"] = full["result"]
    return row


def _json_numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def cmd_mlx_check(args) -> None:
    """Are this machine's simulators the ones the data was made with? Every gold point of the split,
    solved again and compared with the stored result."""
    import platform
    split, workers = args.split, args.workers
    jobs = [(c["tool"], p["args"], p["result"]) for c in load_split(split) for p in c["points"]]
    pool, t0 = ToolPool(workers), time.time()
    try:
        got = pool.call_many([(tool, args) for tool, args, _ in jobs])
    finally:
        pool.close()
    same, worst = 0, 0.0
    for (_, _, want), have in zip(jobs, got):
        if have == want:
            same += 1
            continue
        try:
            w, h = dict(_json_numbers(json.loads(want))), dict(_json_numbers(json.loads(have)))
            rel = max(abs(w[k] - h[k]) / max(abs(w[k]), 1e-12) for k in w) if set(w) == set(h) else float("inf")
        except ValueError:
            rel = float("inf")
        worst = max(worst, rel)
    out = {"split": split, "points": len(jobs), "identical_text": same, "worst_rel_diff": worst,
           "seconds": time.time() - t0, "machine": platform.machine(), "system": platform.system()}
    (HERE / f"{PREFIX}_mlx_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


MLX_STATE_FIELDS = ("ids", "policy", "turns", "turn_starts", "calls", "final_text", "truncated", "unanswered",
                    "answer_closed")


def cmd_mlx_probe(args) -> None:
    """``probe`` for a machine without CUDA: the same episodes, scored and summarised the same way, with the
    generation step done by mlx-lm one episode at a time (greedy). Rows carry each call's result text."""
    from mlx_lm import load
    from transformers import AutoTokenizer
    import mlx.core as mx
    model, _ = load(args.model)
    if args.f32:
        model.set_dtype(mx.float32)
    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, padding_side="left")
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    im_end, eot = hf_tok.convert_tokens_to_ids("<|im_end|>"), hf_tok.convert_tokens_to_ids("<|endoftext|>")
    prefix = generation_prefix(hf_tok)
    cases = load_split(args.split)
    if args.limit:
        cases = random.Random(0).sample(cases, args.limit)
    out_dir = RUNS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}-T0" + ("", "-stop", "-stopans")[args.stop_at_json]
    reuse = {}
    if args.reuse:
        assert args.stop_at_json == 2, "--reuse is for the answer-object rule"
        for line in open(args.reuse):
            d = json.loads(line)
            if not d["answer_closed"] or extract_answer(d["final_text"])[0] is not None:
                reuse[d["id"]] = d
    check = set(random.Random(0).sample(sorted(reuse), min(args.verify, len(reuse))))
    reused = {"reused": 0, "generated": 0, "verified_identical": 0, "verified_different": []}
    pool = ToolPool(args.workers)
    cache: dict = {}
    episodes, scores, t0 = [], [], time.time()
    prompts = [hf_tok(render_prompt(hf_tok, c), add_special_tokens=False).input_ids for c in cases]
    base = mlx_shared_prefix(model, prompts) if args.prompt_cache else None
    if base:
        print(f"shared prefix {len(base[1])} of {min(map(len, prompts))}-{max(map(len, prompts))} prompt tokens "
              f"cached in {time.time() - t0:.0f}s", flush=True)
    state_path = out_dir / f"{tag}.state.jsonl"
    done = [json.loads(line) for line in state_path.open()] if state_path.exists() else []
    assert [d["id"] for d in done] == [c["id"] for c in cases[:len(done)]], "state file is from another run"
    seconds_before = done[-1]["seconds"] if done else 0.0
    for d, case, ids in zip(done, cases, prompts):
        ep = Episode(case, ids)
        for k in MLX_STATE_FIELDS:
            setattr(ep, k, d[k])
        episodes.append(ep)
        scores.append(score_episode(ep))
    if done:
        print(f"resumed: {len(done)} episodes from {state_path.name}", flush=True)
    try:
        with (out_dir / f"{tag}.episodes.jsonl").open("a" if done else "w") as fh, state_path.open("a") as st:
            for i, case in enumerate(cases, 1):
                if i <= len(done):
                    continue
                old = reuse.get(case["id"])
                if old is not None and case["id"] not in check:
                    ep = Episode(case, prompts[i - 1])
                    for k in MLX_STATE_FIELDS:
                        setattr(ep, k, old[k])
                    reused["reused"] += 1
                else:
                    ep = mlx_rollout(model, hf_tok, case, pool, max_new_tokens=args.max_new_tokens,
                                     stop_at_json=args.stop_at_json, cache=cache, prefix=prefix, im_end=im_end, eot=eot,
                                     base=base)
                    reused["generated"] += 1
                    if old is not None:
                        same = ep.ids == old["ids"]
                        reused["verified_identical"] += same
                        if not same:
                            reused["verified_different"].append(case["id"])
                        print(f"verify {case['id']}: {'identical' if same else 'DIFFERENT'} ({len(ep.ids)} tokens)", flush=True)
                sc = score_episode(ep)
                episodes.append(ep)
                scores.append(sc)
                fh.write(json.dumps(mlx_episode_row(ep, sc)) + "\n")
                fh.flush()
                st.write(json.dumps({"id": case["id"], "seconds": seconds_before + time.time() - t0}
                                    | {k: getattr(ep, k) for k in MLX_STATE_FIELDS}) + "\n")
                st.flush()
                if i % 8 == 0 or i == len(cases):
                    print(f"{i}/{len(cases)}  reward so far {sum(s['total'] for s in scores) / i:.3f}  "
                          f"{time.time() - t0:.0f}s  solves {pool.solved}  | {MLX_CLOCK['turns']} turns: read "
                          f"{MLX_CLOCK['prefill_tokens']} tok {MLX_CLOCK['prefill_s']:.0f}s, wrote {MLX_CLOCK['new_tokens']} tok "
                          f"{MLX_CLOCK['generate_s']:.0f}s, copies {MLX_CLOCK['copy_s']:.0f}s, tools {MLX_CLOCK['tool_s']:.0f}s",
                          flush=True)
    finally:
        pool.close()
    summary = summarise(episodes, scores)
    summary["meta"] = {"model": args.model, "backend": "mlx-lm", "split": args.split, "temperature": 0.0, "samples": 1,
                       "max_new_tokens": args.max_new_tokens, "stop_at_json": bool(args.stop_at_json),
                       "stop_rule": ("none", "first top-level {...} of a call-free turn",
                                     "answer object of a call-free turn")[args.stop_at_json],
                       "reuse": (reused | {"from": args.reuse}) if args.reuse else None,
                       "seconds": seconds_before + time.time() - t0, "solves": pool.solved, "limit": args.limit,
                       "prompt_cache": bool(args.prompt_cache), "float32": bool(args.f32), "clock": MLX_CLOCK}
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["overall"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump-tools").set_defaults(fn=cmd_dump_tools)
    p = sub.add_parser("generate")
    p.add_argument("--split", default="all", choices=["all", *SPLITS])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cases per split (0 = the full size)")
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("check")
    p.add_argument("--decontam", type=int, default=1)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("elasticity")
    p.add_argument("--n", type=int, default=12, help="scenarios per tool")
    p.set_defaults(fn=cmd_elasticity)
    p = sub.add_parser("check-harness")
    p.add_argument("--compare", type=int, default=1, help="0: the direction-sentences-only ablation (no check_limits)")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_check_harness)
    p = sub.add_parser("probe")
    p.add_argument("--compare", type=int, default=1, help="0: the direction-sentences-only ablation (no check_limits)")
    p.add_argument("--split", default="dev")
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", default="raw")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=3072)  # 06: see PPO_DEFAULTS
    p.add_argument("--stop-at-json", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.add_argument("--swap", choices=["shared", "holdout"], default=None,
                   help="restate every limit with 05's first sentence or holdout_shift's unseen one (a diagnostic)")
    p.add_argument("--tools-json", default=None,
                   help="declare the tools from this file instead (e.g. 05's, to probe its seed under its own prompt)")
    p.add_argument("--forced", type=int, default=0,
                   help="1: only the questions with a recovery episode, each started from its slipped turns (a diagnostic)")
    p.add_argument("--merge", type=int, default=0, help="1: fold the adapter into the weights for generation (07's probes)")
    p.add_argument("--generator", choices=["torch", "mlx"], default="torch",
                   help="08: mlx -- generated by mlx-lm with the adapter folded in (PPO_DEFAULTS 'generator')")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("reference")
    p.add_argument("--compare", type=int, default=1, help="0: the direction-sentences-only ablation (no check_limits)")
    p.add_argument("--split", default="train")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--recover", type=int, default=0, help="1: the split's recovery episodes (07), not the blind references")
    p.set_defaults(fn=cmd_reference)
    p = sub.add_parser("seed")
    p.add_argument("--compare", type=int, default=1, help="0: the direction-sentences-only ablation (no check_limits)")
    p.add_argument("--out", required=True)
    p.add_argument("--label", choices=["full", "scaffold"], required=True)
    p.add_argument("--reasoning", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--recover", type=int, default=0,
                   help="add this many recovery episodes of other questions to the first --limit reference episodes (07)")
    p.add_argument("--recovery-every", type=int, default=0,
                   help="08: the other 07's recipe (effa2e4): every k-th of the first --limit replaced by its own "
                        "question's recovery episode (RT_RECOVERY)")
    p.add_argument("--fix-lines", type=int, default=0,
                   help="08: with --recovery-every, 07's fix line opens each recovery episode's fix (with_fix_line)")
    p.add_argument("--label-per-class", type=int, default=0,
                   help="label the decision on this many train records per (family, answer class)")
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.set_defaults(fn=cmd_seed)
    p = sub.add_parser("ppo")
    p.add_argument("--compare", type=int, default=1, help="0: the direction-sentences-only ablation (no check_limits)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.add_argument("--resume", type=int, default=0, help="1: continue from <out>/last/")
    for k, v in PPO_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=None)
    p.set_defaults(fn=cmd_ppo)
    p = sub.add_parser("critic-fit", help="08: the candidate critics compared on 07's PPO episodes, and the prover's K")
    p.add_argument("--candidates", default="policy,qwen3-1.7b,qwen3.5-2b,qwen3-4b",
                   help=f"comma-separated, of: policy (07's critic's features), {', '.join(CRITICS)}")
    p.add_argument("--train-run", default="07_recover_from_errors_runs/ppo-n200-r100-turn-s0",
                   help="a PPO run that sampled every step (its samples.jsonl: train questions)")
    p.add_argument("--heldout", default="07_recover_from_errors_runs/seed-n200-r100/epoch1/dev-T1-n120-stop.episodes.jsonl,"
                                        "07_recover_from_errors_runs/seed-n300-r0/epoch1/dev-T1-n120-stop.episodes.jsonl",
                   help="comma-separated probe episode files (dev questions)")
    p.add_argument("--policy-adapter", default="07_recover_from_errors_runs/seed-n200-r100/epoch1",
                   help="the adapter of the 'policy' candidate (07's seed: the PPO run's start)")
    p.add_argument("--max-step", type=int, default=0, help="only the PPO run's steps before this (0: all)")
    p.add_argument("--window", type=int, default=25, help="online replay: steps the heads are fitted on (PPO's)")
    p.add_argument("--stride", type=int, default=2, help="online replay: test every n-th step")
    p.add_argument("--folds", type=int, default=4, help="cross-validation folds, by question")
    p.add_argument("--tolerance", type=float, default=0.005, help="nats within which the cheaper candidate is chosen")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--features-only", type=int, default=0, help="1: compute and cache the features, fit nothing")
    p.add_argument("--cached-only", type=int, default=0, help="1: fit only candidates whose features are cached")
    p.set_defaults(fn=cmd_critic_fit)
    p = sub.add_parser("alive", help="judge a dev probe: every behaviour sampled and every answer class right at least once")
    p.add_argument("episodes", help="a probe's .episodes.jsonl")
    p.add_argument("--split", default="dev")
    p.set_defaults(fn=cmd_alive)
    p = sub.add_parser("device-bench", help="07: the pieces of a PPO step on this machine, and the MPS delta rule checked")
    p.add_argument("--episode", type=int, default=1, help="which train reference episode (06's file order)")
    p.set_defaults(fn=cmd_device_bench)
    sub.add_parser("worker").set_defaults(fn=cmd_worker)
    p = sub.add_parser("xcheck")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=12)
    p.set_defaults(fn=cmd_xcheck)
    p = sub.add_parser("mlx-check")
    p.add_argument("--split", default="dev")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_mlx_check)
    p = sub.add_parser("mlx-probe")
    p.add_argument("--model", required=True, help="HF id or a local MLX model directory")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=3072)  # 06: see PPO_DEFAULTS
    p.add_argument("--stop-at-json", type=int, default=2, choices=[0, 1, 2],
                   help="0 no rule, 1 probe's (first {...}), 2 the answer object only")
    p.add_argument("--reuse", default=None, help="state file of a --stop-at-json 1 run of the same model and split")
    p.add_argument("--verify", type=int, default=0, help="regenerate this many reused episodes and compare tokens")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id (default: the model's)")
    p.add_argument("--prompt-cache", type=int, default=1)
    p.add_argument("--f32", type=int, default=1, help="float32 activations; the weights stay as stored")
    p.set_defaults(fn=cmd_mlx_probe)
    args = ap.parse_args()
    global COMPARE_ON
    COMPARE_ON = bool(getattr(args, "compare", 1))
    args.fn(args)


if __name__ == "__main__":
    main()
