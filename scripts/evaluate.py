"""Evaluate predictions in OUT/<clip>/ against labels in labels/. Same as `egovision eval`.

    python scripts/evaluate.py --pred outputs --labels labels [--json outputs/eval_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from egovision.evaluate import evaluate_dir, format_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, default=Path("outputs"))
    ap.add_argument("--labels", type=Path, default=Path("labels"))
    ap.add_argument("--json", type=Path, default=None, help="Also write the full report as JSON")
    a = ap.parse_args()
    rep = evaluate_dir(a.pred, a.labels)
    print(format_report(rep))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
