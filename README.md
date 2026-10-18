# SafetyNetBench

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 RL environment in which an agent works a
queue of synthetic SNAP (food assistance) applications under the FFY 2026 federal rules. It records
decisions and sends notices, and is scored exactly, including harm metrics: eligible families wrongly
denied, and food aid withheld.

- Environment, rules and metrics: [environments/safety_net_bench/README.md](environments/safety_net_bench/README.md)
- On the Prime Intellect Hub: `bencurtis2003/safety-net-bench`

```bash
uv pip install -e environments/safety_net_bench
pytest tests   # offline checks, no model needed
```

All applicants are synthetic. This is a benchmark ruleset, not eligibility advice. MIT license.
