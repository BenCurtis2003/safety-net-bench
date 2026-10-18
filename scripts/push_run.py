"""Upload a finished local run (traces.jsonl) to your private Prime Intellect Evaluations tab.

Usage: python scripts/push_run.py <run-dir> [--name NAME]
Mirrors what `vf-eval --push` does at the end of a run, for runs made with --no-push.
"""

import argparse
import json
import pathlib

import prime_runs as pr
from verifiers.v1.episode import Episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.run_dir)
    cfg = json.load(open(d / "configs" / "resolved" / "eval.json"))
    episodes = [Episode.model_validate_json(line) for line in open(d / "traces.jsonl") if line.strip()]
    tasks = {e.task.data.idx for e in episodes} if hasattr(episodes[0].task, "data") else set()
    run = pr.init(
        mode="online",
        name=a.name or cfg.get("run", {}).get("name") or d.name,
        environments=["caseworker"],
        model=cfg.get("model"),
        framework="verifiers",
        config={"model": cfg.get("model"), "num_examples": len(tasks) or len(episodes),
                "rollouts_per_example": cfg.get("num_rollouts", 1)},
    )
    run.log_episodes(episodes)
    run.finish(pr.metrics.from_episodes(episodes))
    print(f"{len(episodes)} episodes -> {run.url}  status={run.status.value}")
    if run.errors or any(run.failed_records.values()):
        print("upload issues:", run.errors, dict(run.failed_records))


if __name__ == "__main__":
    main()
