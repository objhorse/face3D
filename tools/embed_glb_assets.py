"""Replace named placeholders in an HTML viewer with base64 GLB assets."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="PLACEHOLDER=PATH",
    )
    args = parser.parse_args()

    html = args.html.read_text(encoding="utf-8")
    for specification in args.asset:
        placeholder, separator, raw_path = specification.partition("=")
        if not separator or not placeholder or not raw_path:
            raise ValueError(f"invalid asset specification: {specification}")
        path = Path(raw_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        token = f"__{placeholder}__"
        if token not in html:
            raise ValueError(f"placeholder {token} is missing from {args.html}")
        html = html.replace(token, encoded)

    args.html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
