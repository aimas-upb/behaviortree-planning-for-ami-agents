"""Concatenate all .txt ontology files into a single prompt file with XML tags."""

import os
from pathlib import Path

ONTOLOGIES_DIR = Path(__file__).resolve().parent.parent / "ontologies" / "txt"
OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "ontologies"
    / "ontologies-prompt.txt"
)


def main():
    parts = []
    for filepath in sorted(ONTOLOGIES_DIR.glob("*.txt")):
        name = filepath.stem
        content = filepath.read_text(encoding="utf-8")
        parts.append(f'<ontology name="{name}">\n{content}</ontology>')

    result = "\n".join(parts)
    OUTPUT_FILE.write_text(result, encoding="utf-8")
    print(f"Wrote {len(parts)} ontologies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
