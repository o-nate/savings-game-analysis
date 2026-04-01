"""Remove cells tagged `solution` from an instructor Jupyter notebook (for student handouts)."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def strip_solution_cells(nb_path: Path) -> nbformat.NotebookNode:
    nb = nbformat.read(nb_path, as_version=4)
    kept = []
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags") or []
        if "solution" in tags:
            continue
        kept.append(cell)
    nb.cells = kept
    return nb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write a copy of the notebook without cells tagged 'solution'."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "savings_game_analysis.ipynb",
        help="Instructor notebook path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "notebooks"
        / "savings_game_analysis_student.ipynb",
        help="Student notebook path.",
    )
    args = parser.parse_args(argv)

    nb = strip_solution_cells(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output.resolve())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
