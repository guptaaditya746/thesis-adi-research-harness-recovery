#!/usr/bin/env python3
"""Create a clearly labelled successful presentation run without model calls."""

from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.demo.curated_success import create_curated_success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", default="doc_448795ae1e2f4823")
    args = parser.parse_args()
    run = create_curated_success(Settings(), args.document_id)
    print(json.dumps({
        "run_id": run.id,
        "status": run.status,
        "accepted_triples": len(run.final_triples),
        "question": run.question,
        "ui": "http://localhost:3000",
    }, indent=2))


if __name__ == "__main__":
    main()

