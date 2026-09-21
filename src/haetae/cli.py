"""Local command-line interface for a verified shared-v1 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import create_app
from .runtime import HaetaeRuntimeError, load_runtime


MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _read_request(path: str) -> dict:
    if path == "-":
        data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    else:
        data = Path(path).read_bytes()
    if len(data) > MAX_REQUEST_BYTES:
        raise HaetaeRuntimeError("request exceeds the 8 MiB input limit")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HaetaeRuntimeError("request is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {"state", "questions"}:
        raise HaetaeRuntimeError("request fields must be state and questions")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haetae")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide = subparsers.add_parser("decide")
    decide.add_argument("--bundle", required=True)
    decide.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    decide.add_argument("--input", default="-")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--bundle", required=True)
    serve.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        if args.command == "decide":
            request = _read_request(args.input)
            runtime = load_runtime(args.bundle, device=args.device)
            decisions = runtime.decide(request["state"], request["questions"])
            output = {
                "model": "haetae-shared-v1",
                "run_id": runtime.bundle.manifest["model"]["run_id"],
                "generation": runtime.bundle.manifest["model"]["generation"],
                "decisions": decisions,
                "calibrated": False,
            }
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
            return 0

        if not 1 <= args.port <= 65535:
            raise HaetaeRuntimeError("port must be between 1 and 65535")
        import uvicorn

        uvicorn.run(
            create_app(args.bundle, device=args.device),
            host=args.host,
            port=args.port,
        )
        return 0
    except (HaetaeRuntimeError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
