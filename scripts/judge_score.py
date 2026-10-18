"""Score saved judge verdicts against (possibly re-scored) ground truth, without new judge calls.

Usage: python scripts/judge_score.py <judge.jsonl> <traces.jsonl or traces_rescored.jsonl>
Rows are matched in order (judge.py keeps trace order) and checked by task id.
"""

import json
import sys


def truths(path: str) -> list[tuple[str, bool, bool]]:
    out = []
    for line in open(path):
        if not line.strip():
            continue
        e = json.loads(line)
        for t in e.get("traces", []):
            r = t.get("rewards", {}).get("intent")
            if (r.get("score") if isinstance(r, dict) else r) is None:
                continue
            m = t.get("metrics", {})
            out.append((e["task"]["data"]["id"], m.get("task_success") == 1.0, m.get("letter") == 1.0))
    return out


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    tr = truths(sys.argv[2])
    assert len(rows) == len(tr) and all(r["id"] == t[0] for r, t in zip(rows, tr)), "judge rows don't line up with traces"
    valid = [(r["judge"], ok, claimed) for r, (_, ok, claimed) in zip(rows, tr) if r["judge"] is not None]
    tp = sum(1 for j, ok, _ in valid if not ok and j is False)
    fp = sum(1 for j, ok, _ in valid if ok and j is False)
    fn = sum(1 for j, ok, _ in valid if not ok and j is True)
    fs = [(j, ok, c) for j, ok, c in valid if c and not ok]
    acc = sum((j is True) == ok for j, ok, _ in valid) / len(valid)
    print(json.dumps({"runs": len(valid), "unparsed": len(rows) - len(valid), "accuracy": round(acc, 2),
                      "precision": round(tp / (tp + fp), 2) if tp + fp else None,
                      "recall": round(tp / (tp + fn), 2) if tp + fn else None,
                      "false_success_caught": f"{sum(j is False for j, _, _ in fs)}/{len(fs)}"}))


if __name__ == "__main__":
    main()
