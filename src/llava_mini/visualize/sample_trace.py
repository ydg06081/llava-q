from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def build_demo_trace() -> dict[str, Any]:
    return {
        "image": "data/toy/images/red_square.png",
        "prompt_tokens": ["<|im_start|>", "user", "<image>", "What", "color", "?"],
        "answer_tokens": ["The", "square", "is", "red", "."],
        "labels": [-100, -100, -100, -100, -100, -100, 785, 9201, 374, 2513, 13],
        "shapes": {
            "pixel_values": [1, 3, 336, 336],
            "clip_patch_features": [1, 576, 1024],
            "projected": [1, 576, 896],
            "spliced_inputs_embeds": [1, 581, 896],
        },
    }


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
    parser.add_argument("--trace-json", help="Path to a trace JSON file.")
    parser.add_argument("--demo", action="store_true", help="Render a built-in demonstration trace.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    args = parser.parse_args()

    if args.demo:
        trace = build_demo_trace()
    elif args.trace_json:
        trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    else:
        parser.error("Provide either --demo or --trace-json.")
    path = write_trace_html(trace, args.out)
    print(f"Wrote trace HTML: {path}")


if __name__ == "__main__":
    main()
