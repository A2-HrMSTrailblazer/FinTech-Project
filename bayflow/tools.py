"""Tools the agent can call, with their permission tier.

Permission tiers (enforced here in code, not in the prompt):
  READ     analyse records and explain results           -> agent may call
  SUGGEST  create a pending correction for the owner     -> agent may call
  APPROVE  apply a correction to the records             -> owner only
  NEVER    move money, contact anyone, apply for credit  -> no such tool exists

Tool output is wrapped and labelled as data. Text inside the owner's records
(for example a purchase note) is never treated as an instruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import checks
from .data import CORRECTION_KINDS, DataError, Dataset, describe_current_value
from .store import Store

READ, SUGGEST, APPROVE = "READ", "SUGGEST", "APPROVE"
AGENT_TIERS = {READ, SUGGEST}
MAX_LIST_ITEMS = 25


class PermissionDenied(Exception):
    pass


@dataclass
class Tool:
    name: str
    tier: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


def _cap(obj, n=MAX_LIST_ITEMS):
    """Keep tool output small: long lists are cut and the cut is stated."""
    if isinstance(obj, dict):
        return {k: _cap(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [_cap(v, n) for v in obj[:n]]
        if len(obj) > n:
            out.append({"_truncated": f"{len(obj) - n} more items not shown"})
        return out
    return obj


def _forecast_view(ds: Dataset, days: int = 31) -> dict:
    f = checks.forecast_cash(ds, days)
    f.pop("days")
    f.pop("bills_at_risk")
    return f


def _cash_flow_view(ds: Dataset, month: str | None = None, compare_to: str | None = None) -> dict:
    c = checks.cash_flow_summary(ds, month, compare_to)
    for k in ("month", "previous_month"):
        if k in c:
            c[k].pop("paid_by_category", None)
    return c


def _propose(ds: Dataset, store: Store, session: str, kind: str, args: dict, reason: str) -> dict:
    if kind not in CORRECTION_KINDS:
        raise ValueError(f"kind must be one of {sorted(CORRECTION_KINDS)}")
    missing = [k for k in CORRECTION_KINDS[kind] if k not in args]
    if missing:
        raise ValueError(f"args missing: {missing}")
    args = {k: args[k] for k in CORRECTION_KINDS[kind]}
    current = describe_current_value(ds, kind, args)
    pid = store.add_proposal(kind, args, current, reason, session, ds.dataset_id)
    return {"proposal_id": pid, "status": "pending",
            "message": "Saved for the owner to approve or reject. Nothing in the records has changed."}


def build_tools() -> dict[str, Tool]:
    date_s = {"type": "string", "description": "Date as YYYY-MM-DD"}
    month_s = {"type": "string", "description": "Month as YYYY-MM"}
    days_s = {"type": "integer", "description": "Forecast length in days, 1 to 180 (default 31)"}
    tools = [
        Tool("reconcile_sales_payments", READ,
             "Check whether recorded sales were actually received: matches QR bills to QR/bank receipts and "
             "cash sales to the daily cash count. Flags missing, late and double-entered payments. "
             "Defaults to the latest month.",
             {"type": "object", "properties": {"start": date_s, "end": date_s}},
             lambda ds, **a: checks.reconcile_sales_payments(ds, a.get("start"), a.get("end"))),
        Tool("cash_flow_summary", READ,
             "Cash flow and profit for a month compared with another month: opening cash, money received, money "
             "paid, closing cash, net sales, gross and operating profit, expense-to-sales ratios per category and "
             "percentage changes. Includes the formulas used. Defaults to the latest month vs the month before.",
             {"type": "object", "properties": {"month": month_s, "compare_to": month_s}},
             lambda ds, **a: _cash_flow_view(ds, a.get("month"), a.get("compare_to"))),
        Tool("forecast_cash", READ,
             "Project cash day by day and list every upcoming bill with the cash left after paying it and a status "
             "(ok / below buffer / cannot pay). States the forecast period; nothing after it is checked.",
             {"type": "object", "properties": {"days": days_s}},
             lambda ds, **a: _forecast_view(ds, int(a.get("days", 31)))),
        Tool("check_spending", READ,
             "Answer 'can I afford this?': compares the forecast with and without a planned purchase, lists bills "
             "that become hard to pay (amount, due date), and compares financing options (demo options are "
             "fictional), including how many repayments fall after the forecast period.",
             {"type": "object", "properties": {
                 "amount": {"type": "number"}, "spend_date": date_s, "description": {"type": "string"},
                 "compare_financing": {"type": "boolean"}, "horizon_days": days_s},
              "required": ["amount", "spend_date"]},
             lambda ds, **a: checks.check_spending(ds, float(a["amount"]), a["spend_date"],
                                                   a.get("description", "Planned purchase"),
                                                   bool(a.get("compare_financing", True)),
                                                   int(a.get("horizon_days", 31)))),
        Tool("check_tax_documents", READ,
             "List purchases from VAT-registered suppliers that have no tax invoice (input VAT at risk), and "
             "estimate the month's VAT payment and PP.30 due date. Planning estimate only.",
             {"type": "object", "properties": {"month": month_s}},
             lambda ds, **a: checks.check_tax_documents(ds, a.get("month"))),
        Tool("propose_correction", SUGGEST,
             "Create a correction for the OWNER to approve. It does not change any record. Kinds and their args: "
             "void_bill{bill_id}; set_tax_invoice{purchase_id, has_tax_invoice}; set_cash_count{date, counted_cash}.",
             {"type": "object", "properties": {
                 "kind": {"type": "string", "enum": sorted(CORRECTION_KINDS)},
                 "args": {"type": "object", "description": "Arguments for the chosen kind"},
                 "reason": {"type": "string", "description": "Why, citing the record IDs"}},
              "required": ["kind", "args", "reason"]},
             None),  # needs the store; handled in Dispatcher
    ]
    return {t.name: t for t in tools}


class Dispatcher:
    """Runs tool calls for one actor and enforces the permission tiers."""

    def __init__(self, ds: Dataset, store: Store, session: str = "demo", actor: str = "agent"):
        self.ds, self.store, self.session, self.actor = ds, store, session, actor
        self.tools = build_tools()

    def declarations(self) -> list[dict]:
        return [{"name": t.name, "description": f"[{t.tier}] {t.description}", "parameters": t.parameters}
                for t in self.tools.values() if t.tier in AGENT_TIERS]

    def call(self, name: str, args: dict | None) -> dict:
        args = dict(args or {})
        tool = self.tools.get(name)
        if tool is None or (self.actor == "agent" and tool.tier not in AGENT_TIERS):
            self.store.log(self.actor, "tool_refused", name, tool.tier if tool else None, args,
                           "not permitted for this actor", self.session)
            raise PermissionDenied(f"Tool '{name}' is not available to the {self.actor}.")
        try:
            if tool.name == "propose_correction":
                result = _propose(self.ds, self.store, self.session, args.get("kind"), args.get("args") or {},
                                  str(args.get("reason", ""))[:500])
                self.store.log(self.actor, "proposal", name, tool.tier, args, f"proposal {result['proposal_id']}",
                               self.session)
            else:
                result = tool.fn(self.ds, **args)
                self.store.log(self.actor, "tool_call", name, tool.tier, args, "ok", self.session)
        except (ValueError, KeyError, TypeError, DataError) as e:
            self.store.log(self.actor, "tool_call", name, tool.tier, args, f"error: {e}", self.session)
            return {"error": str(e)}
        return _cap(result)


def owner_decide(store: Store, proposal_id: int, approve: bool, owner_name: str = "owner") -> dict:
    """The only path that approves a correction. Called from the owner's UI, never by the agent."""
    p = store.decide(proposal_id, approve, owner_name)
    store.log("owner", "decision", "owner_decide", APPROVE, {"proposal_id": proposal_id, "approve": approve},
              p["status"])
    return p
