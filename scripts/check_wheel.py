"""Install a wheel into an isolated target and verify packaged schemas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise ValueError(f"wheel not found: {wheel}")
    with tempfile.TemporaryDirectory(prefix="paper-repro-wheel-") as directory:
        target = Path(directory) / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        package = target / "paper_reproduction_lab"
        schemas = sorted((package / "schemas").glob("*.json"))
        if len(schemas) != 3:
            raise ValueError(f"expected three packaged schemas, found {len(schemas)}")
        for schema in schemas:
            payload = json.loads(schema.read_text(encoding="utf-8"))
            if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                raise ValueError(f"invalid schema: {schema.name}")
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(target)!r}); "
            "import paper_reproduction_lab as package; "
            "assert package.__version__ == '0.1.0'; "
            "assert str(package.__file__).startswith(sys.path[0])"
        )
        subprocess.run([sys.executable, "-c", code], cwd=directory, check=True)
    print("clean wheel install loaded package and three Draft 2020-12 schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
