"""SQLite store for the audit log and the approval queue.

The audit log is append-only from the application's point of view: every tool
call (allowed or refused), every proposal and every owner decision is written
here. Exported, it is the competition's sandbox operation log.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session TEXT,
            actor TEXT NOT NULL,          -- agent | owner | system
            event TEXT NOT NULL,          -- tool_call | tool_refused | proposal | decision | message | number_check
            tool TEXT,
            tier TEXT,
            args TEXT,
            outcome TEXT
        );
        CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts TEXT NOT NULL,
            session TEXT,
            dataset TEXT NOT NULL DEFAULT 'demo',
            kind TEXT NOT NULL,
            args TEXT NOT NULL,
            current_value TEXT,
            reason TEXT,
            status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
            decided_ts TEXT,
            decided_by TEXT
        );
        """)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(proposals)")}
        if "dataset" not in cols:          # database made by an earlier version
            self.db.execute("ALTER TABLE proposals ADD COLUMN dataset TEXT NOT NULL DEFAULT 'demo'")
        self.db.commit()

    # ---- audit -------------------------------------------------------------
    def log(self, actor: str, event: str, tool: str | None = None, tier: str | None = None,
            args: dict | None = None, outcome: str | None = None, session: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO audit_log (ts, session, actor, event, tool, tier, args, outcome) VALUES (?,?,?,?,?,?,?,?)",
            (_now(), session, actor, event, tool, tier, json.dumps(args, default=str) if args is not None else None,
             outcome))
        self.db.commit()

    def audit_rows(self, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM audit_log ORDER BY id" + (f" DESC LIMIT {int(limit)}" if limit else "")
        return [dict(r) for r in self.db.execute(q)]

    # ---- proposals ---------------------------------------------------------
    def add_proposal(self, kind: str, args: dict, current_value, reason: str, session: str | None = None,
                     dataset: str = "demo") -> int:
        cur = self.db.execute(
            "INSERT INTO proposals (created_ts, session, dataset, kind, args, current_value, reason) "
            "VALUES (?,?,?,?,?,?,?)",
            (_now(), session, dataset, kind, json.dumps(args), json.dumps(current_value, default=str), reason))
        self.db.commit()
        return int(cur.lastrowid)

    def proposals(self, status: str | None = None, dataset: str | None = None) -> list[dict]:
        where, p = [], []
        if status:
            where.append("status = ?"); p.append(status)
        if dataset:
            where.append("dataset = ?"); p.append(dataset)
        q = "SELECT * FROM proposals" + (" WHERE " + " AND ".join(where) if where else "")
        rows = []
        for r in self.db.execute(q + " ORDER BY id", p):
            d = dict(r)
            d["args"] = json.loads(d["args"])
            d["current_value"] = json.loads(d["current_value"]) if d["current_value"] else None
            rows.append(d)
        return rows

    def decide(self, proposal_id: int, approve: bool, decided_by: str) -> dict:
        row = self.db.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(f"No proposal {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(f"Proposal {proposal_id} was already {row['status']}")
        status = "approved" if approve else "rejected"
        self.db.execute("UPDATE proposals SET status = ?, decided_ts = ?, decided_by = ? WHERE id = ?",
                        (status, _now(), decided_by, proposal_id))
        self.db.commit()
        return next(p for p in self.proposals() if p["id"] == proposal_id)

    def approved_corrections(self, dataset: str = "demo") -> list[dict]:
        return [{"kind": p["kind"], "args": p["args"]} for p in self.proposals("approved", dataset)]
