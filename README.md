# caseworker

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 RL environment in which an agent works a
queue of synthetic SNAP (food assistance) applications under the FFY 2026 federal rules. It records
decisions and sends notices, and is scored exactly, including harm metrics: eligible families wrongly
denied, and food aid withheld.

- Environment, rules and metrics: [environments/caseworker/README.md](environments/caseworker/README.md)
- On the Prime Intellect Hub: `bencurtis2003/caseworker`

```bash
uv pip install -e environments/caseworker
pytest tests   # offline checks, no model needed
```

All applicants are synthetic. This is a benchmark ruleset, not eligibility advice. MIT license.
