"""Curves for a training run, drawn against the lines that make them readable.

A reward curve on its own says almost nothing. This experiment already knows
what several strategies that do not solve the task are worth, and a rising
number only means something once it is placed against them:

* **0.086** — the frozen 0.5B baseline. Where the policy started.
* **0.245** — `baselines.constant`: valid JSON, the same guess every time.
  A curve below this line is worse than answering without reading the record.
* **0.890** — `baselines.skip_correction`: every step right except the
  temperature correction. Reaching here would mean the arithmetic is being done.

Held-out evaluations are overlaid as points wherever a run directory contains
them, because the gap between the training reward and the held-out reward is the
thing this project exists to look at. `runs/ppo-ac-ie2-s0` is the case in point:
held-out reward rose 0.086 -> 0.132 while `cause_acc` stayed at 0.145, which is
1/7 and therefore chance. The reward moved; the diagnosis did not.

The panels beyond reward are there because reward alone cannot distinguish
learning from collapse. `runs/gonogo-stage-s0` reached a *training* reward of
1.000 while its completion length fell from 106 tokens to 35 and its held-out
reward was 0.000. On the reward panel that run looks like a triumph until step
190; on the length and diversity panels it looks like what it was.

    python3 plots.py runs/ppo-ac-ie2-s0
    python3 plots.py runs/a runs/b --labels main,probe --out compare.png
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


def load_metrics(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics.jsonl in {run}")
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
            # Held-out evaluations, placed at the end of their run.
            for i, (run, rows) in enumerate(zip(runs, series)):
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
        if log:
            ax.set_yscale("log")
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
        for name, overall in load_evals(run):
            print(
                f"  held-out [{name}]  reward {overall.get('reward', 0):.4f}  "
                f"cause {overall.get('cause_acc', 0):.3f}  flags {overall.get('flags_acc', 0):.3f}  "
                f"EM {overall.get('exact_match', 0):.3f}"
            )


if __name__ == "__main__":
    main()
