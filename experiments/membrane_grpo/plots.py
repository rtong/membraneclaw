"""Curves for a training run, drawn against the lines that make them readable.

A reward curve on its own says almost nothing. This experiment already knows
what several strategies that do not solve the task are worth, and a rising
number only means something once it is placed against them:

* **0.086** — the frozen 0.5B baseline. Where the policy started.
* **0.245** — `baselines.constant`: valid JSON, the same guess every time.
  A curve below this line is worse than answering without reading the record.
* **0.890** — `baselines.skip_correction`: every step right except the
  temperature correction. Reaching here would mean the arithmetic is being done.

`grpo_scratch.py` writes a held-out evaluation to `eval.jsonl` as it trains, and
that curve is drawn against the training reward on the same axis. The gap
between the two is what this project exists to look at, so it should be hard to
miss and hard to read as anything else.

The panels beyond reward exist because reward alone cannot tell learning from
collapse. Both failure modes have already been observed on this task in the
sibling actor-critic experiment (`../notebooks/smoke_test`): one run held a
training reward of 1.000 from step 75 while its completion length fell 106 -> 35
tokens and its held-out reward was 0.000, and the length and entropy panels
showed it about 85 steps before the reward panel did.

    python3 plots.py runs/grpo-main-s0
    python3 plots.py runs/grpo-main-s0 runs/grpo-probe-s0 --labels main,probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# Reference lines, from baselines.py. Recomputing them here would let the two
# drift apart, so they are imported.
try:
    from reward import MAIN  # noqa: F401  (import proves the module is reachable)

    FROZEN_BASELINE = 0.086
    CONSTANT_GUESS = 0.245
    SKIP_CORRECTION = 0.890
except ImportError:  # pragma: no cover - plotting should not require the package
    FROZEN_BASELINE, CONSTANT_GUESS, SKIP_CORRECTION = 0.086, 0.245, 0.890

REFERENCE_LINES = (
    (FROZEN_BASELINE, "frozen baseline", "#888888"),
    (CONSTANT_GUESS, "constant guess", "#b8860b"),
    (SKIP_CORRECTION, "skip correction", "#2e7d32"),
)

#: Each panel: (metric key, title, whether a log scale suits it).
PANELS = (
    ("reward_mean", "training reward", False),
    ("completion_tokens", "completion length (tokens)", False),
    ("unique_completions", "unique completions per group", False),
    ("adv_zero_frac", "groups with zero advantage", False),
    ("grad_norm", "gradient norm", True),
    ("entropy_proxy", "entropy proxy (-mean logp)", False),
)

#: Held-out series worth their own panel. `cause_acc` is the one to watch: at
#: 1/7 = 0.143 it is chance, and a run whose reward climbs while this stays flat
#: has improved something other than the diagnosis.
EVAL_PANEL = ("cause_acc", "flags_acc", "validity_gate", "exact_match")
CHANCE_CAUSE = 1.0 / 7.0


def load_metrics(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics.jsonl in {run}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_eval_curve(run: Path) -> list[dict[str, Any]]:
    """Periodic held-out evaluations written by the training loop."""
    path = run / "eval.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_evals(run: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every held-out evaluation found under a run directory."""
    found = []
    for path in sorted(run.glob("**/eval_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "overall" in payload:
            found.append((path.parent.name, payload["overall"]))
    return found


def smooth(values: list[float], window: int) -> list[float]:
    """Centred moving average. The raw series is drawn underneath it."""
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", default=None, help="comma-separated, one per run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=9)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Path(r) for r in args.runs]
    labels = (
        [s.strip() for s in args.labels.split(",")] if args.labels else [r.name for r in runs]
    )
    if len(labels) != len(runs):
        raise SystemExit(f"{len(labels)} labels for {len(runs)} runs")

    series = [load_metrics(run) for run in runs]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, title, log) in zip(axes.flat, PANELS):
        drew = False
        for i, (rows, label) in enumerate(zip(series, labels)):
            present = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
            if not present:
                continue
            drew = True
            steps = [s for s, _ in present]
            values = [float(v) for _, v in present]
            colour = colours[i % len(colours)]
            ax.plot(steps, values, alpha=0.25, linewidth=1, color=colour)
            ax.plot(
                steps,
                smooth(values, args.smooth),
                linewidth=2,
                color=colour,
                label=label if key == "reward_mean" else None,
            )

        if key == "reward_mean":
            for value, name, colour in REFERENCE_LINES:
                ax.axhline(value, linestyle="--", linewidth=1, color=colour, alpha=0.8)
                ax.annotate(
                    f"{name} {value:.3f}",
                    xy=(0.01, value),
                    xycoords=("axes fraction", "data"),
                    fontsize=7,
                    color=colour,
                    va="bottom",
                )
            # The held-out curve, if the run evaluated as it went. Training
            # reward and held-out reward on one axis is the entire picture:
            # the gap between them is what "reward rose" has to be read against.
            for i, run in enumerate(runs):
                curve = load_eval_curve(run)
                if not curve:
                    continue
                colour = colours[i % len(colours)]
                ax.plot(
                    [r["step"] for r in curve],
                    [r["reward"] for r in curve],
                    marker="o",
                    markersize=3,
                    linestyle=":",
                    linewidth=1.6,
                    color=colour,
                    label=f"{labels[i]} held-out",
                )

            # One-off evaluations sitting in a run directory.
            for i, (run, rows) in enumerate(zip(runs, series)):
                if load_eval_curve(run):
                    continue
                last_step = rows[-1]["step"] if rows else 0
                for name, overall in load_evals(run):
                    ax.scatter(
                        [last_step],
                        [overall.get("reward", 0.0)],
                        marker="D",
                        s=45,
                        zorder=5,
                        color=colours[i % len(colours)],
                        edgecolor="black",
                        linewidth=0.6,
                    )
                    ax.annotate(
                        f"held-out {overall.get('reward', 0):.3f}\ncause {overall.get('cause_acc', 0):.3f}",
                        xy=(last_step, overall.get("reward", 0.0)),
                        fontsize=7,
                        ha="right",
                        va="top",
                    )
            ax.legend(fontsize=8, loc="upper left")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.25)
        if log and any(
            v > 0 for rows in series for v in [r.get(key) or 0 for r in rows]
        ):
            ax.set_yscale("log")
        if key == "adv_zero_frac" and any(load_eval_curve(r) for r in runs):
            # Repurpose this panel when held-out detail is available: the
            # component breakdown says more than the degenerate-group count.
            ax.clear()
            for i, run in enumerate(runs):
                curve = load_eval_curve(run)
                for j, metric in enumerate(EVAL_PANEL):
                    if not any(metric in r for r in curve):
                        continue
                    ax.plot(
                        [r["step"] for r in curve],
                        [r.get(metric, 0.0) for r in curve],
                        linestyle=["-", "--", ":", "-."][j % 4],
                        linewidth=1.5,
                        color=colours[i % len(colours)],
                        label=f"{labels[i]} {metric}" if len(runs) > 1 else metric,
                    )
            ax.axhline(CHANCE_CAUSE, linestyle="--", linewidth=1, color="#888888")
            ax.annotate(
                f"chance {CHANCE_CAUSE:.3f}",
                xy=(0.01, CHANCE_CAUSE),
                xycoords=("axes fraction", "data"),
                fontsize=7,
                color="#888888",
                va="bottom",
            )
            ax.set_title("held-out components", fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            drew = True
        if not drew:
            ax.text(0.5, 0.5, f"no {key}", ha="center", va="center", fontsize=9, alpha=0.5)

    fig.suptitle(args.title or " vs ".join(labels), fontsize=12)
    fig.tight_layout()

    out = args.out or (runs[0] / "curves.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # A text summary too, so the numbers are quotable without opening the image.
    for run, rows, label in zip(runs, series, labels):
        rewards = [r["reward_mean"] for r in rows if "reward_mean" in r]
        tokens = [r.get("completion_tokens", 0) for r in rows]
        print(f"\n{label}: {len(rows)} steps")
        if rewards:
            print(
                f"  training reward  first {rewards[0]:.4f}  max {max(rewards):.4f}  "
                f"last {rewards[-1]:.4f}"
            )
        if any(tokens):
            print(f"  completion tokens  first {tokens[0]:.0f}  last {tokens[-1]:.0f}")
        curve = load_eval_curve(run)
        if curve:
            first, last = curve[0], curve[-1]
            print(
                f"  held-out reward  step {first['step']} {first['reward']:.4f}"
                f"  ->  step {last['step']} {last['reward']:.4f}"
            )
            print(
                f"  held-out cause   step {first['step']} {first.get('cause_acc', 0):.3f}"
                f"  ->  step {last['step']} {last.get('cause_acc', 0):.3f}"
                f"   (chance {CHANCE_CAUSE:.3f})"
            )
        for name, overall in load_evals(run):
            print(
                f"  held-out [{name}]  reward {overall.get('reward', 0):.4f}  "
                f"cause {overall.get('cause_acc', 0):.3f}  flags {overall.get('flags_acc', 0):.3f}  "
                f"EM {overall.get('exact_match', 0):.3f}"
            )


if __name__ == "__main__":
    main()
