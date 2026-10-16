"""Deterministic checks. Every number BayFlow shows comes from here.

The language model never calculates anything. It calls these functions and
explains the result. Findings carry the record IDs they came from and results
carry the formulas used, so the owner can check the working themselves.
"""
from __future__ import annotations

import bisect
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .data import Dataset

TAX_RULES = json.loads((Path(__file__).resolve().parent.parent / "config" / "tax_rules.json").read_text())

QR_FAST_WINDOW_MIN = 10        # a QR payment normally lands within minutes
QR_LATE_WINDOW_HOURS = 36      # after this the payment is treated as missing
LATE_AFTER_HOURS = 2
CASH_TOLERANCE_THB = 20        # small rounding / change differences are ignored
DUPLICATE_WINDOW_SEC = 120
NON_OPERATING = {"vat_payment", "owner_drawings"}   # cash out, but not a business cost
FLEXIBLE = {"owner_drawings"}                       # the owner can choose to take less

FORMULAS = {
    "closing_cash": "opening cash + money received - money paid",
    "net_sales": "recorded sales x 100 / 107 (VAT removed; VAT-registered shop)",
    "ingredient_cost": "ingredient purchases - input VAT that can be claimed (purchases with a tax invoice)",
    "gross_profit": "net sales - ingredient cost",
    "operating_profit": "gross profit - operating expenses (owner drawings and VAT payments are not costs)",
    "expense_to_sales_pct": "expense / net sales x 100",
    "change_pct": "(current - previous) / previous x 100",
}


def _r(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def _pct(part: float, whole: float) -> float | None:
    return None if not whole else _r(part / whole * 100)


def _change_pct(cur: float, prev: float) -> float | None:
    return None if not prev else _r((cur - prev) / abs(prev) * 100)


def _period(ds: Dataset, start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(end) if end else ds.as_of
    start_ts = pd.Timestamp(start) if start else end_ts.replace(day=1)
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    return start_ts.normalize(), end_ts.normalize()


def _vat_registered(ds: Dataset) -> bool:
    return str(ds.settings.get("vat_registered", "yes")).lower() != "no"


# ---------------------------------------------------------------------------
# 1. Did we receive the money recorded as sales?
# ---------------------------------------------------------------------------
def _bills(ds: Dataset) -> pd.DataFrame:
    s = ds.sales.copy()
    s["sig"] = s["item"].astype(str) + "x" + s["qty"].astype(str) + "=" + s["line_total"].astype(str)
    b = s.groupby("bill_id").agg(timestamp=("timestamp", "min"), method=("payment_method", "first"),
                                 total=("line_total", "sum"),
                                 sig=("sig", lambda x: "|".join(sorted(x)))).reset_index()
    b["date"] = b.timestamp.dt.normalize()
    return b.sort_values("timestamp").reset_index(drop=True)


def _twin_bills(bills: pd.DataFrame) -> dict[str, str]:
    """Bills identical to the bill just before them (same items, same payment
    method, within two minutes). Returns {later_bill: earlier_bill}."""
    twins = {}
    for _, grp in bills.groupby(["sig", "method"]):
        prev = None
        for row in grp.sort_values("timestamp").itertuples():
            if prev is not None and (row.timestamp - prev.timestamp).total_seconds() <= DUPLICATE_WINDOW_SEC:
                twins[row.bill_id] = prev.bill_id
            prev = row
    return twins


def reconcile_sales_payments(ds: Dataset, start: str | None = None, end: str | None = None) -> dict:
    start_ts, end_ts = _period(ds, start, end)
    bills = _bills(ds)
    in_period = bills[(bills.date >= start_ts) & (bills.date <= end_ts)]
    twins = _twin_bills(bills)

    # --- QR: match each QR bill to one bank receipt (same amount, closest in time) ---
    qr_bills = bills[bills.method == "qr"]
    txns = ds.qr_receipts.sort_values("timestamp")
    used: set[str] = set()
    match: dict[str, tuple[str, pd.Timestamp]] = {}
    by_amount: dict[float, list[tuple[pd.Timestamp, str]]] = {}
    for t in txns.itertuples():
        by_amount.setdefault(float(t.amount), []).append((t.timestamp, t.txn_id))
    keys = {a: [ts for ts, _ in lst] for a, lst in by_amount.items()}

    def find(bill_ts, total, lo, hi, nearest):
        lst = by_amount.get(float(total))
        if not lst:
            return None
        i = bisect.bisect_left(keys[float(total)], bill_ts + lo)
        best = None
        while i < len(lst) and lst[i][0] <= bill_ts + hi:
            ts, tid = lst[i]
            if tid not in used:
                if not nearest:
                    return tid, ts
                if best is None or abs(ts - bill_ts) < abs(best[1] - bill_ts):
                    best = (tid, ts)
            i += 1
        return best

    qr_list = list(qr_bills[["bill_id", "timestamp", "total"]].itertuples(index=False))
    for pass_no, (lo, hi) in enumerate([(pd.Timedelta(minutes=-2), pd.Timedelta(minutes=QR_FAST_WINDOW_MIN)),
                                        (pd.Timedelta(minutes=-2), pd.Timedelta(hours=QR_LATE_WINDOW_HOURS))]):
        for bill_id, ts, total in qr_list:
            if bill_id in match:
                continue
            found = find(ts, total, lo, hi, nearest=(pass_no == 0))
            if found is not None:
                used.add(found[0])
                match[bill_id] = found

    issues = []
    qr_p = qr_bills[(qr_bills.date >= start_ts) & (qr_bills.date <= end_ts)]
    for bill in qr_p.itertuples():
        if bill.bill_id not in match:
            if bill.bill_id in twins:
                issues.append({
                    "type": "possible_double_entry", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                    "evidence": {"bill_id": bill.bill_id, "identical_to_bill": twins[bill.bill_id]},
                    "explanation": "This QR bill has no matching bank receipt and is identical to the bill just before it. It may have been keyed in twice.",
                    "suggested_action": "Check the two bills. If it is a double entry, propose voiding the second one."})
            else:
                issues.append({
                    "type": "qr_payment_not_found", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                    "evidence": {"bill_id": bill.bill_id, "bill_time": str(bill.timestamp)},
                    "explanation": f"No QR/bank receipt of this amount arrived within {QR_LATE_WINDOW_HOURS} hours of the bill.",
                    "suggested_action": "Check the bank app and the customer's slip. The payment may have failed or gone to another account."})
            continue
        txn_id, ts = match[bill.bill_id]
        delay_h = (ts - bill.timestamp).total_seconds() / 3600
        if delay_h > LATE_AFTER_HOURS:
            issues.append({
                "type": "qr_payment_late", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                "evidence": {"bill_id": bill.bill_id, "txn_id": txn_id, "received": str(ts)},
                "explanation": f"Payment arrived {delay_h:.1f} hours after the bill. It is not missing, but it is not in that day's bank total.",
                "suggested_action": "No action needed. Shown so the daily totals make sense."})

    unmatched = txns[~txns.txn_id.isin(used) & (txns.timestamp.dt.normalize() >= start_ts)
                     & (txns.timestamp.dt.normalize() <= end_ts)]
    for t in unmatched.itertuples():
        issues.append({
            "type": "receipt_without_bill", "date": t.timestamp.date().isoformat(), "amount": _r(t.amount),
            "evidence": {"txn_id": t.txn_id},
            "explanation": "Money arrived but no QR bill of this amount was recorded.",
            "suggested_action": "Check whether a sale was not entered."})

    # --- cash: compare each day's cash sales with the counted drawer ---
    cash_bills = bills[bills.method == "cash"]
    cash_sales = cash_bills.groupby("date").total.sum()
    counts = ds.cash_counts.set_index("date").counted_cash
    for d in pd.date_range(start_ts, end_ts, freq="D"):
        expected = float(cash_sales.get(d, 0.0))
        if d not in counts.index:
            if expected > 0:
                issues.append({"type": "cash_count_missing", "date": d.date().isoformat(), "amount": _r(expected),
                               "evidence": {"cash_expected": _r(expected)},
                               "explanation": "No cash count was recorded for this day, so cash sales cannot be checked.",
                               "suggested_action": "Please add the day's cash count."})
            continue
        gap = float(counts[d]) - expected
        evidence = {"cash_expected": _r(expected), "cash_counted": _r(counts[d])}
        if gap <= -CASH_TOLERANCE_THB:
            day_ids = set(cash_bills[cash_bills.date == d].bill_id)
            twin_match = [b for b in twins if b in day_ids
                          and abs(float(bills.loc[bills.bill_id == b, "total"].iloc[0]) + gap) < 1]
            if twin_match:
                b = twin_match[0]
                issues.append({
                    "type": "possible_double_entry", "date": d.date().isoformat(), "amount": _r(-gap),
                    "evidence": {"bill_id": b, "identical_to_bill": twins[b], **evidence},
                    "explanation": "The cash drawer is short by exactly the amount of a bill that is identical to the bill just before it. The sale was probably keyed in twice, so no money is actually missing.",
                    "suggested_action": "Check the two bills. If it is a double entry, propose voiding the second one."})
            else:
                issues.append({
                    "type": "cash_short", "date": d.date().isoformat(), "amount": _r(-gap), "evidence": evidence,
                    "explanation": "Counted cash is lower than recorded cash sales. Common causes are wrong change, a counting mistake, or cash used for a purchase without a record.",
                    "suggested_action": "Check the day's cash purchases and the count sheet. Do not assume anyone took money."})
        elif gap >= CASH_TOLERANCE_THB:
            issues.append({
                "type": "cash_over", "date": d.date().isoformat(), "amount": _r(gap), "evidence": evidence,
                "explanation": "Counted cash is higher than recorded cash sales, which usually means a sale was not entered.",
                "suggested_action": "Check for sales missing from the records."})

    recorded = float(in_period.total.sum())
    qr_in = ds.qr_receipts[(ds.qr_receipts.timestamp.dt.normalize() >= start_ts)
                           & (ds.qr_receipts.timestamp.dt.normalize() <= end_ts)]
    qr_received = float(qr_in.amount.sum())
    cash_counted = float(counts[(counts.index >= start_ts) & (counts.index <= end_ts)].sum())
    issues.sort(key=lambda i: (i["date"], i["type"]))
    needs_check = [i for i in issues if i["type"] != "qr_payment_late"]
    return {
        "period": {"start": start_ts.date().isoformat(), "end": end_ts.date().isoformat()},
        "recorded_sales": _r(recorded),
        "qr_received": _r(qr_received),
        "cash_counted": _r(cash_counted),
        "money_received": _r(qr_received + cash_counted),
        "difference": _r(qr_received + cash_counted - recorded),
        "issues_needing_check": len(needs_check),
        "amount_needing_check": _r(sum(i["amount"] for i in needs_check)),
        "issues": issues,
        "note": "Differences can come from timing, keying errors or missed payments. BayFlow does not assume anyone took money.",
    }


# ---------------------------------------------------------------------------
# 2. Cash flow: where money came from and went
# ---------------------------------------------------------------------------
def cash_ledger(ds: Dataset) -> pd.DataFrame:
    """Daily: closing cash = opening cash + money received - money paid."""
    start, end = ds.settings["opening_date"], ds.as_of
    days = pd.date_range(start, end, freq="D")
    qr = ds.qr_receipts.groupby(ds.qr_receipts.timestamp.dt.normalize()).amount.sum()
    cash = ds.cash_counts.groupby("date").counted_cash.sum()
    buys = ds.purchases.groupby("date").amount_incl_vat.sum()
    exp = ds.expenses.pivot_table(index="date", columns="category", values="amount", aggfunc="sum")
    df = pd.DataFrame(index=days)
    df["received_qr"] = qr.reindex(days, fill_value=0.0)
    df["received_cash"] = cash.reindex(days, fill_value=0.0)
    df["paid_ingredients"] = buys.reindex(days, fill_value=0.0)
    for c in exp.columns:
        df[f"paid_{c}"] = exp[c].reindex(days).fillna(0.0)
    paid_cols = [c for c in df.columns if c.startswith("paid_")]
    df["money_received"] = df.received_qr + df.received_cash
    df["money_paid"] = df[paid_cols].sum(axis=1)
    df["closing_cash"] = ds.settings["opening_balance"] + (df.money_received - df.money_paid).cumsum()
    df["opening_cash"] = df.closing_cash - df.money_received + df.money_paid
    df.index.name = "date"
    return df


def closing_balance(ds: Dataset) -> float:
    return float(cash_ledger(ds).closing_cash.iloc[-1])


def _month_figures(ds: Dataset, ledger: pd.DataFrame, month: str) -> dict:
    m = pd.Period(month, freq="M")
    lm = ledger[ledger.index.to_period("M") == m]
    if lm.empty:
        raise ValueError(f"No records for {month}")
    sales = float(ds.sales[ds.sales.timestamp.dt.to_period("M") == m].line_total.sum())
    net_sales = sales * 100 / 107 if _vat_registered(ds) else sales
    p = ds.purchases[ds.purchases.date.dt.to_period("M") == m]
    claimable = float(p[(p.vat_amount > 0) & (p.has_tax_invoice == "yes")].vat_amount.sum()) if _vat_registered(ds) else 0.0
    ingredient_cost = float(p.amount_incl_vat.sum()) - claimable
    e = ds.expenses[ds.expenses.date.dt.to_period("M") == m]
    by_cat = e.groupby("category").amount.sum().astype(float).to_dict()
    opex = {c: v for c, v in by_cat.items() if c not in NON_OPERATING}
    gross = net_sales - ingredient_cost
    operating = gross - sum(opex.values())
    ratios = {"ingredients": _pct(ingredient_cost, net_sales), **{c: _pct(v, net_sales) for c, v in opex.items()}}
    return {
        "month": month,
        "partial_month": bool(lm.index.max() < m.end_time.normalize() or lm.index.min() > m.start_time),
        "opening_cash": _r(lm.opening_cash.iloc[0]),
        "money_received": _r(lm.money_received.sum()),
        "money_paid": _r(lm.money_paid.sum()),
        "closing_cash": _r(lm.closing_cash.iloc[-1]),
        "recorded_sales": _r(sales),
        "net_sales": _r(net_sales),
        "ingredient_cost": _r(ingredient_cost),
        "gross_profit": _r(gross),
        "gross_margin_pct": _pct(gross, net_sales),
        "operating_expenses": {c: _r(v) for c, v in sorted(opex.items())},
        "operating_profit": _r(operating),
        "operating_margin_pct": _pct(operating, net_sales),
        "expense_to_sales_pct": ratios,
        "owner_drawings": _r(by_cat.get("owner_drawings", 0.0)),
        "vat_paid": _r(by_cat.get("vat_payment", 0.0)),
        "paid_by_category": {"ingredient purchases": _r(p.amount_incl_vat.sum()), **{c: _r(v) for c, v in sorted(by_cat.items())}},
    }


def cash_flow_summary(ds: Dataset, month: str | None = None, compare_to: str | None = None) -> dict:
    """Month figures plus a comparison with the month before (or `compare_to`)."""
    month = month or ds.as_of.strftime("%Y-%m")
    compare_to = compare_to or (pd.Period(month, freq="M") - 1).strftime("%Y-%m")
    ledger = cash_ledger(ds)
    cur = _month_figures(ds, ledger, month)
    try:
        prev = _month_figures(ds, ledger, compare_to)
    except ValueError:
        prev = None
    out = {"month": cur, "formulas": FORMULAS,
           "note": "Profit is not the same as cash. Cash also falls when the owner takes money out or pays VAT, "
                   "and ingredient purchases are treated as used in the same month (fresh ingredients)."}
    if prev:
        out["previous_month"] = prev
        keys = ["money_received", "money_paid", "closing_cash", "recorded_sales", "net_sales", "ingredient_cost",
                "gross_profit", "operating_profit", "owner_drawings"]
        out["change_pct"] = {k: _change_pct(cur[k], prev[k]) for k in keys}
        cats = sorted(set(cur["expense_to_sales_pct"]) | set(prev["expense_to_sales_pct"]))
        changes = []
        for c in cats:
            va = prev["ingredient_cost"] if c == "ingredients" else prev["operating_expenses"].get(c, 0.0)
            vb = cur["ingredient_cost"] if c == "ingredients" else cur["operating_expenses"].get(c, 0.0)
            pct = _change_pct(vb, va)
            changes.append({"category": c, "previous": _r(va), "current": _r(vb), "change": _r(vb - va),
                            "change_pct": pct,
                            "share_of_sales_previous_pct": prev["expense_to_sales_pct"].get(c),
                            "share_of_sales_current_pct": cur["expense_to_sales_pct"].get(c),
                            "notable": abs(vb - va) >= 500 and (pct is None or abs(pct) >= 15)})
        changes.sort(key=lambda x: -abs(x["change"]))
        out["cost_changes"] = changes
    return out


# ---------------------------------------------------------------------------
# 3. VAT and tax documents
# ---------------------------------------------------------------------------
def pp30_due_date(tax_month: str) -> date:
    """PP.30 for a tax month is due in the following month."""
    nxt = pd.Period(tax_month, freq="M") + 1
    efile_day = date(nxt.year, nxt.month, TAX_RULES["pp30_due_day_efile"])
    if efile_day <= date.fromisoformat(TAX_RULES["efile_extension_until"]):
        return efile_day
    return date(nxt.year, nxt.month, TAX_RULES["pp30_due_day_paper"])


def vat_estimate(ds: Dataset, month: str) -> dict:
    rate = TAX_RULES["vat_rate_pct"]
    m = pd.Period(month, freq="M")
    sales = float(ds.sales[ds.sales.timestamp.dt.to_period("M") == m].line_total.sum())
    output_vat = sales * rate / (100 + rate)
    p = ds.purchases[ds.purchases.date.dt.to_period("M") == m]
    claimable = float(p[(p.vat_amount > 0) & (p.has_tax_invoice == "yes")].vat_amount.sum())
    return {"month": month, "sales_incl_vat": _r(sales), "output_vat": _r(output_vat),
            "input_vat_claimable": _r(claimable), "estimated_vat_payable": _r(max(output_vat - claimable, 0)),
            "due_date": pp30_due_date(month).isoformat()}


def check_tax_documents(ds: Dataset, month: str | None = None) -> dict:
    month = month or ds.as_of.strftime("%Y-%m")
    m = pd.Period(month, freq="M")
    p = ds.purchases[ds.purchases.date.dt.to_period("M") == m]
    missing = p[(p.vat_amount > 0) & (p.has_tax_invoice != "yes")]
    return {
        "month": month,
        "purchases_missing_tax_invoice": [
            {"purchase_id": r.purchase_id, "date": r.date.date().isoformat(), "supplier": r.supplier,
             "amount_incl_vat": _r(r.amount_incl_vat), "vat_amount": _r(r.vat_amount)} for r in missing.itertuples()],
        "input_vat_at_risk": _r(missing.vat_amount.sum()),
        "vat": vat_estimate(ds, month) if _vat_registered(ds) else None,
        "reminders": [
            "Input VAT can only be claimed with a full tax invoice from the supplier; a normal receipt is not enough.",
            f"Keep tax records for at least {TAX_RULES['record_retention_years']} years.",
        ],
        "disclaimer": "Estimate for planning only. The owner's accountant files the PP.30 return.",
    }


# ---------------------------------------------------------------------------
# 4. Cash forecast and upcoming bill alerts
# ---------------------------------------------------------------------------
def _daily_baseline(ds: Dataset, lookback_days: int = 28) -> tuple[dict, dict]:
    led = cash_ledger(ds)
    led = led[led.index > ds.as_of - pd.Timedelta(days=lookback_days)]
    by_wd = lambda s: s.groupby(s.index.weekday).mean().to_dict()
    return by_wd(led.money_received), by_wd(led.paid_ingredients)


def _scheduled_bills(ds: Dataset, end: pd.Timestamp) -> list[dict]:
    start = ds.as_of + pd.Timedelta(days=1)
    out = []
    for b in ds.upcoming_bills.itertuples():
        due, step = pd.Timestamp(b.due_date), {"weekly": pd.DateOffset(weeks=1), "monthly": pd.DateOffset(months=1)}.get(b.repeat)
        overdue = due < start
        if overdue and step is None:
            out.append({"date": start, "description": f"{b.description} (overdue since {due.date()})",
                        "category": b.category, "amount": float(b.amount)})
            continue
        while due <= end:
            if due >= start:
                out.append({"date": due, "description": b.description, "category": b.category, "amount": float(b.amount)})
            if step is None:
                break
            due = due + step
    has_vat_bill = any(b["category"] == "vat_payment" for b in out)
    if _vat_registered(ds) and not has_vat_bill:          # avoid counting VAT twice
        vat = vat_estimate(ds, ds.as_of.strftime("%Y-%m"))
        m = ds.as_of.to_period("M")
        while True:
            due = pd.Timestamp(pp30_due_date(m.strftime("%Y-%m")))
            if due > end:
                break
            if due >= start:
                label = f"VAT PP.30 for {m.strftime('%B %Y')} (estimate" + \
                        (")" if m == ds.as_of.to_period("M") else f", based on {ds.as_of.strftime('%B')})")
                out.append({"date": due, "description": label, "category": "vat_payment",
                            "amount": vat["estimated_vat_payable"]})
            m += 1
    return sorted(out, key=lambda b: (b["date"], -b["amount"]))


def _status(balance: float, buffer: float) -> str:
    return "cannot pay" if balance < 0 else "below buffer" if balance < buffer else "ok"


def forecast_cash(ds: Dataset, days: int = 31, extra_outflows: list[dict] | None = None) -> dict:
    if not 1 <= days <= 180:
        raise ValueError("days must be between 1 and 180")
    start = ds.as_of + pd.Timedelta(days=1)
    end = ds.as_of + pd.Timedelta(days=days)
    buffer = float(ds.settings["minimum_buffer"])
    balance = start_balance = closing_balance(ds)
    rec, buys = _daily_baseline(ds)
    bills = _scheduled_bills(ds, end)
    for o in extra_outflows or []:
        bills.append({"date": pd.Timestamp(o["date"]), "description": o["description"],
                      "category": o.get("category", "planned"), "amount": float(o["amount"])})
    bills.sort(key=lambda b: (b["date"], b.get("category") != "planned", -b["amount"]))

    rows, alerts = [], []
    lowest, first_below, first_negative = None, None, None
    for k in range(days):
        d = start + pd.Timedelta(days=k)
        r, s = rec.get(d.weekday(), 0.0), buys.get(d.weekday(), 0.0)
        balance += r - s
        paid_today = []
        for b in (b for b in bills if b["date"] == d):
            before = balance
            balance -= b["amount"]
            status = _status(balance, buffer)
            item = {"due_date": d.date().isoformat(), "description": b["description"], "category": b["category"],
                    "amount": _r(b["amount"]), "cash_before": _r(before), "cash_after": _r(balance),
                    "status": status, "can_be_reduced": b["category"] in FLEXIBLE}
            paid_today.append(item)
            if b["category"] != "planned":
                alerts.append(item)
        row = {"date": d.date().isoformat(), "expected_receipts": _r(r), "ingredient_purchases": _r(s),
               "bills": [{"description": p["description"], "amount": p["amount"]} for p in paid_today],
               "closing_cash": _r(balance)}
        rows.append(row)
        if lowest is None or balance < lowest["closing_cash"]:
            lowest = {"date": row["date"], "closing_cash": _r(balance)}
        if first_below is None and balance < buffer:
            first_below = row["date"]
        if first_negative is None and balance < 0:
            first_negative = row["date"]
    return {
        "forecast_period": {"start": start.date().isoformat(), "end": end.date().isoformat(), "days": days},
        "starting_cash": _r(start_balance),
        "minimum_buffer": _r(buffer),
        "lowest_point": lowest,
        "first_day_below_buffer": first_below,
        "first_day_negative": first_negative,
        "bill_alerts": alerts,
        "bills_at_risk": [a for a in alerts if a["status"] != "ok"],
        "days": rows,
        "assumptions": [
            f"Starting cash is the closing balance on {ds.as_of.date()} from the records "
            "(opening cash + money received - money paid).",
            "Daily money received and ingredient purchases = average for the same weekday over the last 28 days.",
            "Bills come from the upcoming bills list, repeated weekly or monthly where marked.",
            "VAT is added on its due date as an estimate unless it is already in the bills list.",
            f"Nothing after {end.date()} is checked.",
        ],
    }


def _monthly_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def _avg_monthly_cash_before_drawings(ds: Dataset) -> tuple[float | None, list[str]]:
    led = cash_ledger(ds)
    months = []
    for m, g in led.groupby(led.index.to_period("M")):
        if g.index.min() == m.start_time.normalize() and g.index.max() == m.end_time.normalize():
            drawn = float(g.get("paid_owner_drawings", pd.Series(0.0)).sum())
            months.append((m.strftime("%Y-%m"), float(g.money_received.sum() - g.money_paid.sum()) + drawn))
    if not months:
        return None, []
    return sum(v for _, v in months) / len(months), [m for m, _ in months]


def _changed_bills(base: list[dict], new: list[dict]) -> list[dict]:
    rank = {"ok": 0, "below buffer": 1, "cannot pay": 2}
    before = {(a["due_date"], a["description"]): a["status"] for a in base}
    out = []
    for a in new:
        was = before.get((a["due_date"], a["description"]), "ok")
        if rank[a["status"]] > rank[was]:
            out.append({**a, "status_without_purchase": was})
    return out


def check_spending(ds: Dataset, amount: float, spend_date: str, description: str = "Planned purchase",
                   compare_financing: bool = True, horizon_days: int = 31) -> dict:
    if amount <= 0:
        raise ValueError("amount must be positive")
    base = forecast_cash(ds, horizon_days)
    period = base["forecast_period"]
    sd = pd.Timestamp(spend_date)
    if not (pd.Timestamp(period["start"]) <= sd <= pd.Timestamp(period["end"])):
        raise ValueError(f"spend_date must be inside the forecast period {period['start']} to {period['end']}")
    buffer = base["minimum_buffer"]
    planned = {"date": spend_date, "description": description, "amount": amount, "category": "planned"}
    cash = forecast_cash(ds, horizon_days, extra_outflows=[planned])
    result = {
        "purchase": {"description": description, "amount": _r(amount), "date": spend_date},
        "forecast_period": period,
        "coverage_note": (f"This check covers {period['start']} to {period['end']} ({period['days']} days) only. "
                          "Staying above the buffer in this period does not show that later repayments are affordable."),
        "minimum_buffer": buffer,
        "without_purchase": {"lowest_point": base["lowest_point"], "bills_at_risk": base["bills_at_risk"]},
        "pay_with_cash": {
            "lowest_point": cash["lowest_point"],
            "first_day_below_buffer": cash["first_day_below_buffer"],
            "first_day_negative": cash["first_day_negative"],
            "stays_above_buffer_in_period": cash["first_day_below_buffer"] is None,
            "bills_made_harder": _changed_bills(base["bill_alerts"], cash["bill_alerts"]),
        },
        "options": [],
    }
    if compare_financing and ds.financing_options:
        avg_free, months_used = _avg_monthly_cash_before_drawings(ds)
        end = pd.Timestamp(period["end"])
        for f in ds.financing_options:
            n = int(f["term_months"])
            pay = _monthly_payment(amount, float(f["annual_rate_pct"]), n)
            dates = [(sd + pd.DateOffset(months=k)).normalize() for k in range(1, n + 1)]
            outs = [{"date": spend_date, "description": f"{f['name']} fee", "amount": float(f["fee_thb"]),
                     "category": "planned"}] if float(f["fee_thb"]) else []
            outs += [{"date": d.date().isoformat(), "description": f"{f['name']} repayment {i}/{n}", "amount": pay,
                      "category": "planned"} for i, d in enumerate(dates, 1) if d <= end]
            fc = forecast_cash(ds, horizon_days, extra_outflows=outs)
            inside = sum(d <= end for d in dates)
            result["options"].append({
                "option_id": f["option_id"], "name": f["name"], "monthly_payment": _r(pay), "term_months": n,
                "total_cost_of_borrowing": _r(pay * n - amount + float(f["fee_thb"])),
                "first_repayment": dates[0].date().isoformat(), "last_repayment": dates[-1].date().isoformat(),
                "repayments_inside_forecast": inside, "repayments_after_forecast": n - inside,
                "lowest_point_in_period": fc["lowest_point"],
                "stays_above_buffer_in_period": fc["first_day_below_buffer"] is None,
                "bills_made_harder": _changed_bills(base["bill_alerts"], fc["bill_alerts"]),
                "payment_share_of_monthly_cash_before_drawings_pct": _pct(pay, avg_free) if avg_free and avg_free > 0 else None,
                "not_checked": (f"{n - inside} of {n} repayments are after {period['end']} and are not in the forecast."
                                if n - inside else None),
            })
        result["options"].sort(key=lambda o: (not o["stays_above_buffer_in_period"], o["total_cost_of_borrowing"]))
        result["long_term_check"] = {
            "average_monthly_cash_before_owner_drawings": None if avg_free is None else _r(avg_free),
            "based_on_months": months_used,
            "explanation": "Rough guide for the months after the forecast: each repayment is compared with the cash the "
                           "business has left in an average full month before the owner takes money out. A repayment "
                           "means the owner can take that much less each month for the whole term.",
        }
        result["financing_note"] = "Financing products in the demo are fictional. Real terms must be checked with the lender."
    return result
