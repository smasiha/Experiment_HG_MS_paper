#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from transformers import AutoConfig, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    try:
        AutoConfig.from_pretrained(args.model, **kwargs)
        AutoTokenizer.from_pretrained(args.model, **kwargs)
    except Exception as exc:
        print(f"MODEL_ACCESS_ERR {args.model}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"MODEL_ACCESS_OK {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
