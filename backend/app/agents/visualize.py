"""
Export the main router graph (router → orders | returns | qna | clarify) for visualization.

- Mermaid: paste into https://mermaid.live or use a Mermaid preview in VS Code / GitHub.
- ASCII: optional `pip install grandalf` then use `draw_ascii()` on the same graph object.
- PNG: `draw_mermaid_png()` may require extra system deps; prefer Mermaid in the browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..graph import get_compiled_graph


def get_main_graph_mermaid() -> str:
    """Return a Mermaid `graph TD` string for the compiled main graph."""
    return get_compiled_graph().get_graph().draw_mermaid()


def get_main_graph_ascii() -> str:
    """Return an ASCII art rendering (requires `pip install grandalf`)."""
    return get_compiled_graph().get_graph().draw_ascii()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export main LangGraph as Mermaid text")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write diagram to this file (UTF-8). If omitted, print to stdout.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Print ASCII diagram instead (needs: pip install grandalf)",
    )
    args = parser.parse_args()
    if args.ascii:
        text = get_main_graph_ascii()
    else:
        text = get_main_graph_mermaid()
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    else:
        print(text)


if __name__ == "__main__":
    main()
