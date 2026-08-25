"""The four-way comparison: what RL moved, and what it did not.

Four policies on the same 200 dev cases, the same prompt version, the same
greedy decoding, scored by the same `eval.py`. That uniformity is the point --
an earlier version of this comparison would have used the 9B numbers from
`prompt_ab.py` over 40 cases, which is a different function on a different
sample, and the bars would not have been measuring the same thing.

The chart is arranged so the flat bar is impossible to miss. `root_cause`
accuracy is **0.145 for all three 0.5B policies** -- identical before training,
after PPO, and after GRPO -- against a 1/7 = 0.143 chance floor. Every other
component moves. That one does not, until the model changes.

    python3 benchmark.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: (label, path, colour). Order is the story: untrained, two RL runs, then the
#: model that can actually do the task.
POLICIES = (
    ("vanilla 0.5B", "runs/baseline-0.5b-v2/eval_dev_greedy.json", "#9e9e9e"),
    (
        "PPO 0.5B",
        "../notebooks/smoke_test/runs/ppo-ac-ie2-s0/eval-dev-greedy/eval_dev_greedy.json",
        "#ef6c00",
    ),
    ("GRPO 0.5B", "runs/trained-main-greedy/eval_dev_greedy.json", "#1565c0"),
    ("vanilla 9B", "runs/vanilla-9b/eval_dev_greedy.json", "#2e7d32"),
)

#: (key, title, reference line or None)
METRICS = (
    ("reward", "reward", 0.245),
    ("cause_acc", "root_cause accuracy", 1.0 / 7.0),
    ("flags_acc", "flags correct", None),
    ("numeric_acc", "numeric correct", None),
    ("schema_ok", "schema valid", None),
    ("exact_match", "exact match", None),
)


def load(path: str) -> dict:
    payload = json.loads((ROOT / path).read_text())
    overall = payload["overall"]
    overall["_n"] = overall["n_cases"]
    overall["_split"] = payload.get("split")
    overall["_prompt"] = payload.get("prompt_version")
    overall["_sha"] = (payload.get("split_sha256") or "")[:12]
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "benchmark.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [(label, load(path), colour) for label, path, colour in POLICIES]

    # Refuse to draw bars that are not the same measurement.
    keys = {(d["_split"], d["_n"], d["_prompt"], d["_sha"]) for _, d, _ in data}
    if len(keys) != 1:
        raise SystemExit(f"policies were not measured comparably: {keys}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    labels = [label for label, _, _ in data]
    colours = [colour for _, _, colour in data]
    x = range(len(data))

    for ax, (key, title, reference) in zip(axes.flat, METRICS):
        values = [d.get(key, 0.0) or 0.0 for _, d, _ in data]
        bars = ax.bar(x, values, color=colours, width=0.62)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        if reference is not None:
            ax.axhline(reference, linestyle="--", linewidth=1, color="#c62828")
            ax.annotate(
                ("chance " if key == "cause_acc" else "constant guess ") + f"{reference:.3f}",
                xy=(0.02, reference),
                xycoords=("axes fraction", "data"),
                fontsize=7,
                color="#c62828",
                va="bottom",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8, rotation=12)
        ax.set_ylim(0, max(1.0, max(values) * 1.25))
        ax.grid(axis="y", alpha=0.25)

    n = data[0][1]["_n"]
    fig.suptitle(
        f"Same {n} dev cases, same prompt {data[0][1]['_prompt']}, same greedy decoding, "
        "same scorer\n"
        "root_cause is flat at chance across all three 0.5B policies; "
        "everything RL moved is format",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    width = max(len(label) for label in labels)
    header = "".join(f"{title[:9]:>10}" for _, title, _ in METRICS)
    print(f"\n{'':<{width}}{header}")
    for label, d, _ in data:
        row = "".join(f"{(d.get(k) or 0.0):>10.3f}" for k, _, _ in METRICS)
        print(f"{label:<{width}}{row}")


if __name__ == "__main__":
    main()
