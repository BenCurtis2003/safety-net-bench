"""Mock case-management tools. All state lives on the rollout's `OfficeState`."""

import json

import verifiers.v1 as vf
from pydantic import Field


class OfficeState(vf.State):
    db: dict[str, dict[str, dict]] = Field(default_factory=dict)
    """table -> record id -> record fields."""
    outbox: list[dict] = Field(default_factory=list)
    """Emails sent during the rollout: {to, subject, body}."""
    report: dict | None = None
    """Final report submitted by the agent: {completed: bool, summary: str}."""
    actions: list[str] = Field(default_factory=list)
    """Audit log of every mutating tool call."""
    flaky: list[str] = Field(default_factory=list)
    """Keys ('table.id' or 'email.addr') whose first write attempt fails transiently."""
    attempts: dict[str, int] = Field(default_factory=dict)
    checks: list[int] = Field(default_factory=list)
    """One entry per `aggregate` call: how many mutating actions had happened at that point."""


def _flaky_hit(state: OfficeState, key: str) -> bool:
    """Fail the first attempt on a flaky key, like a dropped upstream write."""
    if key in state.flaky and state.attempts.get(key, 0) == 0:
        state.attempts[key] = 1
        return True
    return False


class CaseworkerToolset(vf.Toolset[vf.ToolsetConfig, OfficeState]):
    TOOL_PREFIX = None

    @vf.tool
    def list_tables(self) -> str:
        """List the tables in the company database."""
        return json.dumps(sorted(self.state.db))

    @vf.tool
    def list_records(self, table: str, page: int = 1) -> str:
        """Return one page (up to 20 records) of a table as JSON, with paging info.
        Call again with page=2, 3, ... while `has_more` is true."""
        if table not in self.state.db:
            return f"error: no table named {table!r}"
        rows = self.state.db[table]
        keys = sorted(rows)
        size = 20
        start = max(page - 1, 0) * size
        chunk = {k: rows[k] for k in keys[start : start + size]}
        return json.dumps(
            {"page": page, "total_records": len(keys), "has_more": start + size < len(keys), "records": chunk},
            sort_keys=True,
        )

    @vf.tool
    def get_record(self, table: str, record_id: str) -> str:
        """Return one record by id."""
        rows = self.state.db.get(table)
        if rows is None:
            return f"error: no table named {table!r}"
        if record_id not in rows:
            return f"error: no record {record_id!r} in {table!r}"
        return json.dumps(rows[record_id], sort_keys=True)

    @vf.tool
    def update_record(self, table: str, record_id: str, field: str, value: str) -> str:
        """Set one field on one record. Numeric strings are stored as numbers."""
        rows = self.state.db.get(table)
        if rows is None:
            return f"error: no table named {table!r}"
        if record_id not in rows:
            return f"error: no record {record_id!r} in {table!r}"
        if _flaky_hit(self.state, f"{table}.{record_id}"):
            return "error: upstream timeout, the change was NOT saved. Retry the call."
        parsed: object = value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            pass
        rows[record_id][field] = parsed
        self.state.actions.append(f"update {table}.{record_id}.{field}={parsed!r}")
        return "ok"

    @vf.tool
    def delete_record(self, table: str, record_id: str) -> str:
        """Delete one record from a table."""
        rows = self.state.db.get(table)
        if rows is None or record_id not in rows:
            return f"error: no record {record_id!r} in {table!r}"
        del rows[record_id]
        self.state.actions.append(f"delete {table}.{record_id}")
        return "ok"

    @vf.tool
    def create_record(self, table: str, record_id: str, fields_json: str) -> str:
        """Create a new record (fields given as a JSON object). Creates the table if needed."""
        try:
            fields = json.loads(fields_json)
            assert isinstance(fields, dict)
        except Exception:
            return "error: fields_json must be a JSON object"
        rows = self.state.db.setdefault(table, {})
        if record_id in rows:
            return f"error: record {record_id!r} already exists in {table!r}"
        rows[record_id] = fields
        self.state.actions.append(f"create {table}.{record_id}")
        return "ok"

    @vf.tool
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email to one address."""
        if _flaky_hit(self.state, f"email.{to.strip().lower()}"):
            return "error: mail server timeout, the email was NOT sent. Retry the call."
        self.state.outbox.append({"to": to, "subject": subject, "body": body})
        self.state.actions.append(f"email {to}")
        return "sent"

    @vf.tool
    def aggregate(self, table: str, field: str, op: str = "sum", where_field: str = "", where_value: str = "") -> str:
        """Compute over a whole table in one call (no paging). op: sum, count, min, max, avg,
        or missing (how many records lack `field`). Optionally keep only records whose
        `where_field` equals `where_value`. Useful to check your own work before reporting."""
        self.state.checks.append(len(self.state.actions))
        rows = self.state.db.get(table)
        if rows is None:
            return f"error: no table named {table!r}"
        recs = [r for r in rows.values() if not where_field or str(r.get(where_field)) == where_value]
        if op == "missing":
            return json.dumps({"op": op, "records": len(recs), "result": sum(field not in r for r in recs)})
        if op == "count":
            return json.dumps({"op": op, "records": len(recs), "result": sum(field in r for r in recs)})
        vals = []
        for r in recs:
            try:
                vals.append(float(r[field]))
            except (KeyError, TypeError, ValueError):
                pass
        if op not in ("sum", "min", "max", "avg"):
            return "error: op must be one of sum, count, min, max, avg, missing"
        if not vals:
            return json.dumps({"op": op, "records": len(recs), "numeric_values": 0, "result": None})
        res = {"sum": sum(vals), "min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals)}[op]
        return json.dumps({"op": op, "records": len(recs), "numeric_values": len(vals), "result": round(res, 4)})

    @vf.tool
    def submit_report(self, completed: bool, summary: str) -> str:
        """Finish the task. Set `completed` to true only if every requirement was met;
        otherwise set it to false and explain what could not be done."""
        self.state.report = {"completed": completed, "summary": summary}
        return "report received"


if __name__ == "__main__":
    CaseworkerToolset.run()
