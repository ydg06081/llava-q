from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def render_trace_html(trace: dict[str, Any]) -> str:
    prompt_tokens = trace.get("prompt_tokens", [])
    answer_tokens = trace.get("answer_tokens", [])
    labels = trace.get("labels", [])
    shapes = trace.get("shapes", {})

    token_rows = []
    all_tokens = list(prompt_tokens) + list(answer_tokens)
    for idx, token in enumerate(all_tokens):
        label = labels[idx] if idx < len(labels) else ""
        masked = label == -100
        token_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(str(token))}</td>"
            f"<td>{html.escape(str(label))}</td>"
            f"<td>{'masked' if masked else 'loss'}</td>"
            "</tr>"
        )

    shape_items = "".join(
        f"<li><code>{html.escape(str(name))}</code>: {html.escape(str(value))}</li>"
        for name, value in shapes.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLaVA Mini Sample Trace</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f3f3f3; text-align: left; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>LLaVA Mini Sample Trace</h1>
  <p><strong>Image:</strong> {html.escape(str(trace.get("image", "")))}</p>
  <h2>Tensor shapes</h2>
  <ul>{shape_items}</ul>
  <h2>Label masking</h2>
  <table>
    <thead><tr><th>Index</th><th>Token</th><th>Label</th><th>Loss role</th></tr></thead>
    <tbody>{''.join(token_rows)}</tbody>
  </table>
</body>
</html>"""


def write_trace_html(trace: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_trace_html(trace), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True, help="Path to a trace JSON file.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    args = parser.parse_args()

    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    path = write_trace_html(trace, args.out)
    print(f"Wrote trace HTML: {path}")


if __name__ == "__main__":
    main()
