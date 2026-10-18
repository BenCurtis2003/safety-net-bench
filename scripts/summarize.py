"""Summarize SafetyNetBench runs with 95% bootstrap confidence intervals.

Usage: python scripts/summarize.py <run-dir-or-traces.jsonl> [...] [--boot 2000] [--md]

CIs resample *tasks* (all rollouts of a task move together), so repeated rollouts of the
same task don't count as independent evidence. Also reports:
  - pass^k: share of tasks where every rollout succeeded (consistency)
  - per-family success and false success
  - error types among failed close/payroll runs (v0.5+)
  - false success split by whether the agent used the aggregate check tool (v0.5+)
"""

import argparse
import json
import pathlib
import random
from collections import defaultdict


def load(path: str) -> tuple[str, list[dict]]:
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "traces.jsonl"
    rows, model = [], p.parent.name
    cfg = p.parent / "configs" / "resolved" / "eval.json"
    if cfg.exists():
        model = json.load(open(cfg)).get("model", model)
    for line in open(p):
        if not line.strip():
            continue
        e = json.loads(line)
        d = e["task"]["data"]
        for t in e.get("traces", []):
            r = t.get("rewards", {}).get("intent")
            if r is None or t.get("errors"):
                continue  # incomplete rollout: don't score it
            m = t.get("metrics", {})
            rows.append({"task": d["id"], "family": d.get("family"), "m": m,
                         "success": m.get("task_success", r.get("score") if isinstance(r, dict) else r) == 1.0,
                         "claimed": m.get("letter") == 1.0})
    return model, rows


def ci(rows: list[dict], fn, boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% CI of fn(rows), resampling tasks with replacement."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks = list(by_task)
    point = fn(rows)
    if len(tasks) < 2:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(boot):
        sample = [r for t in rng.choices(tasks, k=len(tasks)) for r in by_task[t]]
        v = fn(sample)
        if v == v:
            stats.append(v)
    stats.sort()
    return point, stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def rate(key):
    def f(rows):
        return sum(key(r) for r in rows) / len(rows) if rows else float("nan")
    return f


def fs_given_fail(rows):
    fails = [r for r in rows if not r["success"]]
    return sum(r["claimed"] for r in fails) / len(fails) if fails else float("nan")


def pass_all(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["success"])
    return sum(all(v) for v in by.values()) / len(by) if by else float("nan")


def fmt(x):
    p, lo, hi = x
    if p != p:
        return "-"
    return f"{100 * p:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]" if lo == lo else f"{100 * p:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    for path in a.runs:
        model, rows = load(path)
        tasks = len({r["task"] for r in rows})
        print(f"\n## {model}  ({len(rows)} rollouts, {tasks} tasks, {len(rows) / max(tasks, 1):.1f} per task)\n")
        print("| metric | value [95% CI] |\n|---|---|")
        for name, fn in [
            ("real success (intent)", rate(lambda r: r["success"])),
            ("naive grader pass (claimed done)", rate(lambda r: r["claimed"])),
            ("false success (claimed, not done)", rate(lambda r: r["claimed"] and not r["success"])),
            ("claimed done | failed", fs_given_fail),
            ("honest self-report", rate(lambda r: r["claimed"] == r["success"])),
            ("pass^k (every rollout of a task succeeded)", pass_all),
            ("case accuracy", rate(lambda r: r["m"].get("case_accuracy", float("nan")))),
            ("eligible families wrongly denied (per case)", rate(lambda r: r["m"].get("wrongful_denial_rate", float("nan")))),
            ("eligible families delayed (per case)", rate(lambda r: r["m"].get("eligible_delayed_rate", float("nan")))),
            ("ineligible cases wrongly approved (per case)", rate(lambda r: r["m"].get("wrongful_approval_rate", float("nan")))),
        ]:
            print(f"| {name} | {fmt(ci(rows, fn, a.boot))} |")

        w = [r["m"].get("benefit_withheld_per_case") for r in rows if "benefit_withheld_per_case" in r["m"]]
        if w:
            o = [r["m"].get("benefit_overpaid_per_case", 0) for r in rows]
            print(f"\nfood aid wrongly withheld: ${100 * sum(w) / len(w):,.0f} per 100 cases per month; "
                  f"overpaid: ${100 * sum(o) / len(o):,.0f}")
        print("\n| family | n | success | false success |\n|---|---|---|---|")
        fams = defaultdict(list)
        for r in rows:
            fams[r["family"]].append(r)
        for fam, fr in sorted(fams.items()):
            print(f"| {fam} | {len(fr)} | {fmt(ci(fr, rate(lambda r: r['success']), a.boot))} | "
                  f"{fmt(ci(fr, rate(lambda r: r['claimed'] and not r['success']), a.boot))} |")

        failed = [r for r in rows if not r["success"] and any(k.startswith("err_") for k in r["m"])]
        if failed:
            counts = defaultdict(int)
            for r in failed:
                for k, v in r["m"].items():
                    if k.startswith("err_") and v:
                        counts[k] += 1
            print(f"\nError types in {len(failed)} failed close/payroll runs (a run can show several):\n")
            print("| error type | runs | share |\n|---|---|---|")
            for k, c in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"| {k[4:]} | {c} | {100 * c / len(failed):.0f}% |")

        if any("check_tool_used" in r["m"] for r in rows):
            print("\n| aggregate check tool | runs | success | false success |\n|---|---|---|---|")
            for label, sel in [("used", True), ("not used", False)]:
                g = [r for r in rows if bool(r["m"].get("check_tool_used")) == sel]
                if g:
                    print(f"| {label} | {len(g)} | {fmt(ci(g, rate(lambda r: r['success']), a.boot))} | "
                          f"{fmt(ci(g, rate(lambda r: r['claimed'] and not r['success']), a.boot))} |")
            after = [r for r in rows if r["m"].get("checked_after_last_write")]
            print(f"\nchecked after last write: {len(after)}/{len(rows)} runs")


if __name__ == "__main__":
    main()
