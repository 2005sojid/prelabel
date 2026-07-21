"""One-command launcher.

    python run.py                 # start on 127.0.0.1:8000
    python run.py --port 9000
    python run.py --reload        # auto-reload on code changes (development)

Then open the printed URL in a browser.
"""

from __future__ import annotations

import argparse

import uvicorn

from prelabel import __version__, config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prelabel inference & auto-annotation server")
    parser.add_argument("--host", default=config.HOST, help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=config.PORT, help="port (default: %(default)s)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    banner = f"  Prelabel v{__version__}"
    url = f"  Open: http://{args.host}:{args.port}"
    width = max(len(banner), len(url)) + 4
    print("=" * width)
    print(banner)
    print(url)
    print("=" * width)

    uvicorn.run(
        "prelabel.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
