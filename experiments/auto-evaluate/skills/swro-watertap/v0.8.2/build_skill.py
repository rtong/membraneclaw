from __future__ import annotations

from pathlib import Path


def build(version_dir: Path) -> Path:
    source_dir = version_dir / "source"
    parts = [path.read_text(encoding="utf-8").strip() for path in sorted(source_dir.glob("*.md"))]
    if not parts:
        raise ValueError(f"No source markdown files found in {source_dir}")
    output = version_dir / "SKILL.md"
    output.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    output = build(Path(__file__).resolve().parent)
    print(f"Wrote {output}")
