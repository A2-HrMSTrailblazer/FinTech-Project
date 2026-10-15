"""Deterministic checks. All numbers BayFlow shows come from here.

The language model never calculates anything. It calls these functions and
explains the result. Every finding carries the record IDs it came from, so
the owner can check the working themselves.
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


def _r(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def _period(ds: Dataset, start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    as_of = pd.Timestamp(ds.cash_position["as_of"])
    end_ts = pd.Timestamp(end) if end else as_of
    start_ts = pd.Timestamp(start) if start else end_ts.replace(day=1)
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    return start_ts.normalize(), end_ts.normalize()


def _bills(ds: Dataset) -> pd.DataFrame:
    s = ds.sales.copy()
    s["sig"] = s.item_id + "x" + s.qty.astype(str)
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
        grp = grp.sort_values("timestamp")
        prev = None
        for _, row in grp.iterrows():
            if prev is not None and (row.timestamp - prev.timestamp).total_seconds() <= DUPLICATE_WINDOW_SEC:
                twins[row.bill_id] = prev.bill_id
            prev = row
    return twins


# ---------------------------------------------------------------------------
# 1. Did we receive the money recorded as sales?
# ---------------------------------------------------------------------------
def reconcile_sales_payments(ds: Dataset, start: str | None = None, end: str | None = None) -> dict:
    start_ts, end_ts = _period(ds, start, end)
    bills = _bills(ds)
    in_period = bills[(bills.date >= start_ts) & (bills.date <= end_ts)]
    twins = _twin_bills(bills)

    # --- QR: match each QR bill to one bank receipt ---
    qr_bills = bills[bills.method == "qr"]
    txns = ds.qr_receipts.sort_values("timestamp").copy()
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
    for _, bill in qr_p.iterrows():
        if bill.bill_id not in match:
            if bill.bill_id in twins:
                issues.append({
                    "type": "possible_double_entry", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                    "evidence": {"bill_id": bill.bill_id, "identical_to_bill": twins[bill.bill_id]},
                    "explanation": "This QR bill has no matching bank receipt and is identical to the bill just before it. It may have been keyed in twice.",
                    "suggested_action": "Check the bill. If it is a double entry, propose voiding it."})
            else:
                issues.append({
                    "type": "qr_payment_not_found", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                    "evidence": {"bill_id": bill.bill_id, "bill_time": str(bill.timestamp)},
                    "explanation": f"No PromptPay receipt of this amount arrived within {QR_LATE_WINDOW_HOURS} hours of the bill.",
                    "suggested_action": "Check the bank app and the customer's slip. The payment may have failed or been sent elsewhere."})
            continue
        txn_id, ts = match[bill.bill_id]
        delay_h = (ts - bill.timestamp).total_seconds() / 3600
        if delay_h > LATE_AFTER_HOURS:
            issues.append({
                "type": "qr_payment_late", "date": bill.date.date().isoformat(), "amount": _r(bill.total),
                "evidence": {"bill_id": bill.bill_id, "txn_id": txn_id, "received": str(ts)},
                "explanation": f"Payment arrived {delay_h:.1f} hours after the bill. It is not missing, but it will not show in that day's bank total.",
                "suggested_action": "No action needed. Shown so the daily totals make sense."})

    unmatched_txn = txns[~txns.txn_id.isin(used) & (txns.timestamp.dt.normalize() >= start_ts)
                         & (txns.timestamp.dt.normalize() <= end_ts)]
    for _, t in unmatched_txn.iterrows():
        issues.append({
            "type": "receipt_without_bill", "date": t.timestamp.date().isoformat(), "amount": _r(t.amount),
            "evidence": {"txn_id": t.txn_id},
            "explanation": "Money arrived but no QR bill of this amount was recorded.",
            "suggested_action": "Check whether a sale was not entered in the POS."})

    # --- cash: compare each day's cash sales with the counted drawer ---
    cash_sales = bills[bills.method == "cash"].groupby("date").total.sum()
    counts = ds.cash_counts.set_index("date").counted_cash
    days = pd.date_range(start_ts, end_ts, freq="D")
    for d in days:
        expected = float(cash_sales.get(d, 0.0))
        if d not in counts.index:
            if expected > 0:
                issues.append({"type": "cash_count_missing", "date": d.date().isoformat(), "amount": _r(expected),
                               "evidence": {}, "explanation": "No cash count was recorded for this day.",
                               "suggested_action": "Ask for the day's cash count."})
            continue
        gap = float(counts[d]) - expected
        if gap <= -CASH_TOLERANCE_THB:
            day_twins = [b for b, orig in twins.items()
                         if b in set(bills[(bills.date == d) & (bills.method == "cash")].bill_id)]
            twin_match = [b for b in day_twins if abs(float(bills.loc[bills.bill_id == b, "total"].iloc[0]) + gap) < 1]
            if twin_match:
                b = twin_match[0]
                issues.append({
                    "type": "possible_double_entry", "date": d.date().isoformat(), "amount": _r(-gap),
                    "evidence": {"bill_id": b, "identical_to_bill": twins[b], "cash_expected": _r(expected),
                                 "cash_counted": _r(counts[d])},
                    "explanation": "The cash drawer is short by exactly the amount of a bill that is identical to the bill just before it. The sale was probably keyed in twice, so no money is actually missing.",
                    "suggested_action": "Check the two bills. If it is a double entry, propose voiding the second one."})
            else:
                issues.append({
                    "type": "cash_short", "date": d.date().isoformat(), "amount": _r(-gap),
                    "evidence": {"cash_expected": _r(expected), "cash_counted": _r(counts[d])},
                    "explanation": "Counted cash is lower than recorded cash sales. Common causes are wrong change, a missed count, or cash taken for a purchase without a record.",
                    "suggested_action": "Check the day's cash purchases and the count sheet. Do not assume anyone took money."})
        elif gap >= CASH_TOLERANCE_THB:
            issues.append({
                "type": "cash_over", "date": d.date().isoformat(), "amount": _r(gap),
                "evidence": {"cash_expected": _r(expected), "cash_counted": _r(counts[d])},
                "explanation": "Counted cash is higher than recorded cash sales, which usually means a sale was not entered.",
                "suggested_action": "Check for sales missing from the POS."})

    recorded = float(in_period.total.sum())
    qr_received = float(ds.qr_receipts[(ds.qr_receipts.timestamp.dt.normalize() >= start_ts)
                                       & (ds.qr_receipts.timestamp.dt.normalize() <= end_ts)].amount.sum())
    cash_counted = float(counts[(counts.index >= start_ts) & (counts.index <= end_ts)].sum())
    issues.sort(key=lambda i: (i["date"], i["type"]))
    needs_check = [i for i in issues if i["type"] not in ("qr_payment_late",)]
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
# 2. Stock: expected vs actual ingredient usage
# ---------------------------------------------------------------------------
def _usage_table(ds: Dataset) -> pd.DataFrame:
    s = ds.sales.assign(date=ds.sales.timestamp.dt.normalize())
    exp = s.merge(ds.recipes, on="item_id")
    exp["expected"] = exp.qty * exp.qty_per_item
    expected = exp.groupby(["date", "ingredient_id"]).expected.sum()
    bought = ds.purchases.groupby(["date", "ingredient_id"]).qty.sum()

    sc = ds.stock_counts.sort_values("date")
    rows = []
    for ing, g in sc.groupby("ingredient_id"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            d = g.date[i]
            opening, closing = float(g.closing_count[i - 1]), float(g.closing_count[i])
            purchased = float(bought.get((d, ing), 0.0))
            rows.append({"date": d, "ingredient_id": ing, "opening": opening, "purchased": purchased,
                         "closing": closing, "actual": opening + purchased - closing,
                         "expected": float(expected.get((d, ing), 0.0))})
    u = pd.DataFrame(rows).merge(ds.ingredients, on="ingredient_id")
    u["allowance"] = u.expected * u.waste_allowance_pct / 100
    u["variance"] = u.actual - u.expected
    return u.sort_values(["ingredient_id", "date"]).reset_index(drop=True)


def stock_variance(ds: Dataset, start: str | None = None, end: str | None = None,
                   ingredient_id: str | None = None) -> dict:
    start_ts, end_ts = _period(ds, start, end)
    u = _usage_table(ds)
    if ingredient_id:
        u = u[u.ingredient_id == ingredient_id]
        if u.empty:
            raise ValueError(f"Unknown ingredient_id: {ingredient_id}")
    findings, paired = [], set()
    for ing, g in u.groupby("ingredient_id"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            if not (start_ts <= r.date <= end_ts) or (ing, r.date) in paired:
                continue
            threshold = max(2 * r.allowance, 0.02 * r.expected + (2 if r.unit == "pcs" else 0.1))
            if abs(r.variance) <= threshold:
                continue
            nxt = g.iloc[i + 1] if i + 1 < len(g) else None
            base = {"date": r.date.date().isoformat(), "ingredient_id": ing, "ingredient": r["name"], "unit": r.unit,
                    "expected_usage": _r(r.expected), "actual_usage": _r(r.actual), "difference": _r(r.variance),
                    "evidence": {"opening": _r(r.opening), "purchased": _r(r.purchased), "closing": _r(r.closing)}}
            if nxt is not None and abs(r.variance + nxt.variance) <= max(threshold, abs(r.variance) * 0.25):
                paired.add((ing, nxt.date))
                findings.append({**base, "type": "likely_count_error",
                                 "explanation": f"The difference reverses the next day ({_r(nxt.variance)} {r.unit} on {nxt.date.date()}). The closing count on {r.date.date()} was probably entered wrongly.",
                                 "suggested_action": "Recount if possible, or propose correcting the closing count.",
                                 "value_thb": 0.0})
            elif r.variance > 0:
                value = (r.variance - r.allowance) * r.unit_cost_thb
                findings.append({**base, "type": "unexplained_usage", "value_thb": _r(value),
                                 "explanation": "More was used than the day's sales need, beyond normal waste. Common causes are food served but not billed, spoilage, or larger portions.",
                                 "suggested_action": "Compare with the day's bills and ask staff about that day. Do not assume theft."})
            else:
                findings.append({**base, "type": "usage_below_sales", "value_thb": 0.0,
                                 "explanation": "Less was used than the recorded sales need. A sale may have been recorded twice, or the count is wrong.",
                                 "suggested_action": "Check the day's bills and the count."})
    findings.sort(key=lambda f: (f["date"], f["ingredient_id"]))
    unexplained = [f for f in findings if f["type"] == "unexplained_usage"]
    return {
        "period": {"start": start_ts.date().isoformat(), "end": end_ts.date().isoformat()},
        "ingredients_checked": sorted(u.ingredient_id.unique().tolist()),
        "findings": findings,
        "unexplained_usage_value_thb": _r(sum(f["value_thb"] for f in unexplained)),
        "method": "actual usage = opening + purchased - closing; expected usage = units sold x recipe; normal waste allowance per ingredient is subtracted before valuing a gap.",
    }


# ---------------------------------------------------------------------------
# 3. Why is cash going up or down?
# ---------------------------------------------------------------------------
def _month_flows(ds: Dataset, month: str) -> dict:
    m = pd.Period(month, freq="M")
    sales = float(ds.sales[ds.sales.timestamp.dt.to_period("M") == m].line_total.sum())
    p = ds.purchases[ds.purchases.date.dt.to_period("M") == m]
    e = ds.expenses[ds.expenses.date.dt.to_period("M") == m]
    out = {"ingredients": float(p.amount_incl_vat.sum())}
    out.update(e.groupby("category").amount.sum().astype(float).to_dict())
    return {"sales": sales, "outflows": out}


def explain_cash_change(ds: Dataset, month_a: str, month_b: str) -> dict:
    a, b = _month_flows(ds, month_a), _month_flows(ds, month_b)
    cats = sorted(set(a["outflows"]) | set(b["outflows"]))
    changes = []
    for c in cats:
        va, vb = a["outflows"].get(c, 0.0), b["outflows"].get(c, 0.0)
        delta = vb - va
        pct = (delta / va * 100) if va else None
        changes.append({"category": c, month_a: _r(va), month_b: _r(vb), "change": _r(delta),
                        "change_pct": None if pct is None else _r(pct),
                        "notable": abs(delta) >= 500 and (pct is None or abs(pct) >= 15)})
    changes.sort(key=lambda x: -abs(x["change"]))
    net_a = a["sales"] - sum(a["outflows"].values())
    net_b = b["sales"] - sum(b["outflows"].values())
    return {
        "months": [month_a, month_b],
        "recorded_sales": {month_a: _r(a["sales"]), month_b: _r(b["sales"]),
                           "change": _r(b["sales"] - a["sales"])},
        "total_spending": {month_a: _r(sum(a["outflows"].values())), month_b: _r(sum(b["outflows"].values()))},
        "net_cash_from_operations": {month_a: _r(net_a), month_b: _r(net_b), "change": _r(net_b - net_a)},
        "spending_changes": changes,
        "note": "Based on recorded sales and recorded spending. Cash available is not the same as profit; the owner's own pay and tax are not included.",
    }


# ---------------------------------------------------------------------------
# 4. VAT and tax documents
# ---------------------------------------------------------------------------
def pp30_due_date(tax_month: str) -> date:
    """PP.30 for a tax month is due in the following month."""
    nxt = (pd.Period(tax_month, freq="M") + 1)
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
    month = month or pd.Timestamp(ds.cash_position["as_of"]).strftime("%Y-%m")
    m = pd.Period(month, freq="M")
    p = ds.purchases[ds.purchases.date.dt.to_period("M") == m]
    missing = p[(p.vat_amount > 0) & (p.has_tax_invoice != "yes")]
    return {
        "month": month,
        "purchases_missing_tax_invoice": [
            {"purchase_id": r.purchase_id, "date": r.date.date().isoformat(), "supplier": r.supplier,
             "amount_incl_vat": _r(r.amount_incl_vat), "vat_amount": _r(r.vat_amount)} for r in missing.itertuples()],
        "input_vat_at_risk": _r(missing.vat_amount.sum()),
        "vat": vat_estimate(ds, month),
        "reminders": [
            "Input VAT can only be claimed with a full tax invoice from the supplier; a normal receipt is not enough.",
            f"Keep tax records for at least {TAX_RULES['record_retention_years']} years.",
        ],
        "disclaimer": "Estimate for planning only. The owner's accountant files the PP.30 return.",
    }


# ---------------------------------------------------------------------------
# 5. Cash forecast and spending decisions
# ---------------------------------------------------------------------------
def _daily_baseline(ds: Dataset, lookback_days: int = 28) -> tuple[dict, dict]:
    as_of = pd.Timestamp(ds.cash_position["as_of"])
    lo = as_of - pd.Timedelta(days=lookback_days - 1)
    qr = ds.qr_receipts.assign(date=ds.qr_receipts.timestamp.dt.normalize())
    qr = qr[(qr.date >= lo) & (qr.date <= as_of)].groupby("date").amount.sum()
    cash = ds.cash_counts[(ds.cash_counts.date >= lo) & (ds.cash_counts.date <= as_of)].set_index("date").counted_cash
    days = pd.date_range(lo, as_of)
    receipts = pd.Series([float(qr.get(d, 0)) + float(cash.get(d, 0)) for d in days], index=days)
    buys = ds.purchases[(ds.purchases.date >= lo) & (ds.purchases.date <= as_of)].groupby("date").amount_incl_vat.sum()
    spend = pd.Series([float(buys.get(d, 0)) for d in days], index=days)
    by_wd = lambda s: s.groupby(s.index.weekday).mean().to_dict()
    return by_wd(receipts), by_wd(spend)


def forecast_cash(ds: Dataset, days: int = 31, extra_outflows: list[dict] | None = None,
                  extra_inflows: list[dict] | None = None) -> dict:
    as_of = pd.Timestamp(ds.cash_position["as_of"])
    buffer = float(ds.cash_position.get("minimum_buffer", 0))
    balance = float(ds.cash_position["bank_balance"]) + float(ds.cash_position["cash_on_hand"])
    start_balance = balance
    rec, spend = _daily_baseline(ds)
    vat_month = as_of.strftime("%Y-%m")
    vat = vat_estimate(ds, vat_month)
    bills = [{"date": r.due_date, "description": r.description, "amount": float(r.amount)}
             for r in ds.upcoming_bills.itertuples()]
    bills.append({"date": pd.Timestamp(vat["due_date"]), "description": f"VAT PP.30 for {vat_month} (estimate)",
                  "amount": vat["estimated_vat_payable"]})
    for o in extra_outflows or []:
        bills.append({"date": pd.Timestamp(o["date"]), "description": o["description"], "amount": float(o["amount"])})
    inflows = [{"date": pd.Timestamp(i["date"]), "description": i["description"], "amount": float(i["amount"])}
               for i in extra_inflows or []]

    rows, lowest, first_below, first_negative = [], None, None, None
    for k in range(1, days + 1):
        d = as_of + pd.Timedelta(days=k)
        r, s = rec[d.weekday()], spend[d.weekday()]
        b_today = [b for b in bills if b["date"] == d]
        i_today = [i for i in inflows if i["date"] == d]
        balance += r - s - sum(b["amount"] for b in b_today) + sum(i["amount"] for i in i_today)
        row = {"date": d.date().isoformat(), "expected_receipts": _r(r), "ingredient_spend": _r(s),
               "bills": [{"description": b["description"], "amount": _r(b["amount"])} for b in b_today],
               "other_inflows": [{"description": i["description"], "amount": _r(i["amount"])} for i in i_today],
               "closing_balance": _r(balance)}
        rows.append(row)
        if lowest is None or balance < lowest["closing_balance"]:
            lowest = {"date": row["date"], "closing_balance": _r(balance)}
        if first_below is None and balance < buffer:
            first_below = row["date"]
        if first_negative is None and balance < 0:
            first_negative = row["date"]
    return {
        "as_of": as_of.date().isoformat(),
        "starting_cash": _r(start_balance),
        "minimum_buffer": _r(buffer),
        "lowest_point": lowest,
        "first_day_below_buffer": first_below,
        "first_day_negative": first_negative,
        "days": rows,
        "assumptions": [
            "Daily receipts and ingredient spending = average for the same weekday over the last 28 days of actual money received and purchases.",
            "Known bills from the owner's bill list, plus an estimated VAT payment on its due date.",
            "Does not include the owner's personal withdrawals unless they are entered as bills.",
        ],
    }


def _monthly_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def check_spending(ds: Dataset, amount: float, spend_date: str, description: str = "Planned purchase",
                   compare_financing: bool = True, horizon_days: int = 45) -> dict:
    if amount <= 0:
        raise ValueError("amount must be positive")
    base = forecast_cash(ds, horizon_days)
    with_cash = forecast_cash(ds, horizon_days, extra_outflows=[{"date": spend_date, "description": description,
                                                                 "amount": amount}])
    buffer = base["minimum_buffer"]
    result = {
        "purchase": {"description": description, "amount": _r(amount), "date": spend_date},
        "without_purchase": {"lowest_point": base["lowest_point"]},
        "pay_with_cash": {"lowest_point": with_cash["lowest_point"],
                          "first_day_below_buffer": with_cash["first_day_below_buffer"],
                          "first_day_negative": with_cash["first_day_negative"],
                          "stays_above_buffer": with_cash["first_day_below_buffer"] is None},
        "minimum_buffer": buffer,
        "options": [],
    }
    if compare_financing:
        for f in ds.financing_options:
            pay = _monthly_payment(amount, f["annual_rate_pct"], f["term_months"])
            sd = pd.Timestamp(spend_date)
            outs = [{"date": spend_date, "description": f"{f['name']} fee", "amount": f["fee_thb"]}] if f["fee_thb"] else []
            outs += [{"date": (sd + pd.DateOffset(months=k)).date().isoformat(), "description": f"{f['name']} repayment",
                      "amount": pay} for k in range(1, f["term_months"] + 1)]
            fc = forecast_cash(ds, horizon_days, extra_outflows=outs)
            total_cost = pay * f["term_months"] - amount + f["fee_thb"]
            result["options"].append({
                "option_id": f["option_id"], "name": f["name"], "monthly_payment": _r(pay),
                "term_months": f["term_months"], "total_cost_of_borrowing": _r(total_cost),
                "lowest_point": fc["lowest_point"], "stays_above_buffer": fc["first_day_below_buffer"] is None})
        result["options"].sort(key=lambda o: (not o["stays_above_buffer"], o["total_cost_of_borrowing"]))
        result["financing_note"] = "Financing products in the demo are fictional. Real terms must be checked with the lender."
    return result
