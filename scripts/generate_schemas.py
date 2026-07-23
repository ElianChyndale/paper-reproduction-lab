"""Generate public and packaged Draft 2020-12 schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_reproduction_lab.models import RunManifest, StudyManifest, TrialResult

ROOT = Path(__file__).resolve().parents[1]


def schema_content() -> dict[str, str]:
    outputs: dict[str, str] = {}
    for filename, model in [
        ("study-manifest.schema.json", StudyManifest),
        ("trial-result.schema.json", TrialResult),
        ("run-manifest.schema.json", RunManifest),
    ]:
        schema = model.model_json_schema(mode="validation")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = (
            "https://elianchyndale.github.io/paper-reproduction-lab/" + filename
        )
        outputs[filename] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches: list[str] = []
    for filename, content in schema_content().items():
        for directory in [
            ROOT / "schemas",
            ROOT / "src" / "paper_reproduction_lab" / "schemas",
        ]:
            path = directory / filename
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != content:
                    mismatches.append(path.relative_to(ROOT).as_posix())
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
    if mismatches:
        print("schema mismatch: " + ", ".join(mismatches))
        return 1
    print("schemas are current" if args.check else "schemas generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
