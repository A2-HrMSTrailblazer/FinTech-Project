"""The checks must find every planted problem in the demo data, and nothing else.

The answer key is written by the data generator, not by the checks, so these
tests compare BayFlow against an independent source of truth. Small hand-made
datasets test the cash formulas with numbers worked out on paper.
"""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from bayflow import checks
from bayflow.data import DataError, build_dataset, load_dataset, read_folder, read_uploads, to_excel

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DEMO)


@pytest.fixture(scope="module")
def key():
    return json.loads((DEMO / "answer_key.json").read_text())


@pytest.fixture(scope="module")
def recon(ds):
    return checks.reconcile_sales_payments(ds, "2026-09-01", "2026-09-30")


def tiny(opening=30000, receipts=10000, purchase=20000, bills=15000, upcoming=None):
    """One day of records, small enough to check by hand."""
    raw = {
        "sales": pd.DataFrame([{"bill_id": "B1", "timestamp": "2026-09-01 10:00", "line_total": str(receipts),
                                "payment_method": "qr"}]),
        "qr_receipts": pd.DataFrame([{"txn_id": "T1", "timestamp": "2026-09-01 10:01", "amount": str(receipts)}]),
        "cash_counts": pd.DataFrame([{"date": "2026-09-01", "counted_cash": "0"}]),
        "purchases": pd.DataFrame([{"purchase_id": "P1", "date": "2026-09-01", "supplier": "S",
                                    "amount_incl_vat": str(purchase)}]),
        "expenses": pd.DataFrame([{"date": "2026-09-01", "category": "rent", "amount": str(bills)}]),
        "upcoming_bills": pd.DataFrame(upcoming or [], columns=["due_date", "description", "amount", "repeat"]),
        "settings": pd.DataFrame([{"opening_date": "2026-09-01", "opening_balance": str(opening),
                                   "minimum_buffer": "1000", "vat_registered": "no"}]),
    }
    return build_dataset({k: v.astype(str) for k, v in raw.items()})


# ---- cash formulas ----------------------------------------------------------------
def test_team_worked_example_closing_cash():
    # Team PDF example: 30,000 opening + 10,000 receipt - 20,000 purchase - 15,000 bills.
    # The PDF says -5,000, but the arithmetic gives +5,000.
    led = checks.cash_ledger(tiny())
    assert led.opening_cash.iloc[0] == 30000
    assert led.closing_cash.iloc[-1] == 5000


def test_cash_shortfall_goes_negative():
    # 30,000 + 10,000 - 20,000 - 25,000 = -5,000
    assert checks.cash_ledger(tiny(bills=25000)).closing_cash.iloc[-1] == -5000


def test_closing_cash_matches_answer_key(ds, key):
    assert checks.closing_balance(ds) == pytest.approx(key["closing_balance"])


def test_month_closing_links_to_next_opening(ds):
    s = checks.cash_flow_summary(ds, "2026-09")
    assert s["previous_month"]["closing_cash"] == s["month"]["opening_cash"]
    m = s["month"]
    assert m["closing_cash"] == pytest.approx(m["opening_cash"] + m["money_received"] - m["money_paid"])


def test_profit_formulas(ds):
    m = checks.cash_flow_summary(ds, "2026-09")["month"]
    assert m["net_sales"] == pytest.approx(m["recorded_sales"] * 100 / 107, abs=0.01)
    assert m["gross_profit"] == pytest.approx(m["net_sales"] - m["ingredient_cost"], abs=0.01)
    assert m["operating_profit"] == pytest.approx(m["gross_profit"] - sum(m["operating_expenses"].values()), abs=0.01)
    assert "owner_drawings" not in m["operating_expenses"] and "vat_payment" not in m["operating_expenses"]
    assert m["expense_to_sales_pct"]["rent"] == pytest.approx(12000 / m["net_sales"] * 100, abs=0.01)


def test_explains_electricity_increase(ds, key):
    s = checks.cash_flow_summary(ds, "2026-09", "2026-08")
    elec = next(c for c in s["cost_changes"] if c["category"] == "electricity")
    assert (elec["previous"], elec["current"]) == (key["expense_change"]["august"], key["expense_change"]["september"])
    assert elec["change_pct"] == pytest.approx((6150 - 4200) / 4200 * 100, abs=0.01)
    assert elec["notable"]
    assert not next(c for c in s["cost_changes"] if c["category"] == "rent")["notable"]


# ---- payment matching -----------------------------------------------------------------
def _issues(recon, kind):
    return [i for i in recon["issues"] if i["type"] == kind]


def test_finds_missed_qr_payment(recon, key):
    found = _issues(recon, "qr_payment_not_found")
    assert [(i["evidence"]["bill_id"], i["amount"]) for i in found] == \
        [(k["bill_id"], k["amount"]) for k in key["missed_qr"]]


def test_late_payment_is_not_reported_as_missing(recon, key):
    assert [i["evidence"]["bill_id"] for i in _issues(recon, "qr_payment_late")] == \
        [k["bill_id"] for k in key["late_qr"]]


def test_double_entry_explains_cash_gap(recon, key):
    dup = _issues(recon, "possible_double_entry")
    assert len(dup) == 1
    assert dup[0]["evidence"]["bill_id"] == key["duplicate_bills"][0]["duplicate"]
    assert dup[0]["evidence"]["identical_to_bill"] == key["duplicate_bills"][0]["original"]


def test_cash_shortfall(recon, key):
    assert [(i["date"], i["amount"]) for i in _issues(recon, "cash_short")] == \
        [(k["date"], k["amount"]) for k in key["cash_short"]]


def test_no_other_payment_issues(recon):
    assert len(recon["issues"]) == 4
    assert -recon["difference"] == pytest.approx(recon["amount_needing_check"])


def test_clean_month_has_no_issues(ds):
    assert checks.reconcile_sales_payments(ds, "2026-08-01", "2026-08-31")["issues"] == []


# ---- tax --------------------------------------------------------------------------------
def test_missing_tax_invoices(ds, key):
    t = checks.check_tax_documents(ds, "2026-09")
    assert sorted(p["purchase_id"] for p in t["purchases_missing_tax_invoice"]) == \
        sorted(k["purchase_id"] for k in key["missing_tax_invoice"])
    assert t["input_vat_at_risk"] == pytest.approx(sum(k["vat_amount"] for k in key["missing_tax_invoice"]))
    assert t["vat"]["due_date"] == "2026-10-23"


def test_pp30_falls_back_to_paper_deadline_after_extension():
    assert checks.pp30_due_date("2026-12").isoformat() == "2027-01-23"
    assert checks.pp30_due_date("2027-01").isoformat() == "2027-02-15"


# ---- forecast and bill alerts -------------------------------------------------------------
def test_forecast_states_its_period_and_lists_bills(ds):
    f = checks.forecast_cash(ds, 31)
    assert f["forecast_period"] == {"start": "2026-10-01", "end": "2026-10-31", "days": 31}
    assert f["starting_cash"] == 19000
    descs = [b["description"] for b in f["bill_alerts"]]
    assert "Shop rent" in descs and sum("weekly drawing" in d for d in descs) == 4
    assert any(d.startswith("VAT PP.30 for September") for d in descs)
    assert f["bills_at_risk"] == []


def test_bill_alert_status_by_hand():
    upcoming = [{"due_date": "2026-09-03", "description": "Rent", "amount": "4000", "repeat": "none"},
                {"due_date": "2026-09-04", "description": "Wages", "amount": "1500", "repeat": "none"}]
    ds = tiny(opening=10000, receipts=0, purchase=0, bills=0, upcoming=upcoming)
    f = checks.forecast_cash(ds, 5)
    a = {b["description"]: b for b in f["bill_alerts"]}
    assert a["Rent"]["cash_after"] == 6000 and a["Rent"]["status"] == "ok"
    assert a["Wages"]["cash_after"] == 4500 and a["Wages"]["status"] == "ok"
    f2 = checks.forecast_cash(tiny(opening=5000, receipts=0, purchase=0, bills=0, upcoming=upcoming), 5)
    a2 = {b["description"]: b for b in f2["bill_alerts"]}
    assert a2["Rent"]["status"] == "ok" and a2["Rent"]["cash_after"] == 1000
    assert a2["Wages"]["status"] == "cannot pay" and a2["Wages"]["cash_after"] == -500


def test_vat_is_not_counted_twice():
    upcoming = [{"due_date": "2026-09-23", "description": "VAT", "amount": "700", "repeat": "none"}]
    ds = tiny(upcoming=upcoming)
    ds.settings["vat_registered"] = "yes"
    ds.tables["upcoming_bills"]["category"] = "vat_payment"
    f = checks.forecast_cash(ds, 30)
    assert [b["description"] for b in f["bill_alerts"]] == ["VAT"]


# ---- can I afford this? -------------------------------------------------------------------
def test_cash_purchase_lists_bills_made_harder(ds):
    c = checks.check_spending(ds, 25000, "2026-10-03", "Second chest freezer")
    assert c["forecast_period"]["end"] == "2026-10-31"
    assert "2026-10-01 to 2026-10-31 (31 days) only" in c["coverage_note"]
    pw = c["pay_with_cash"]
    assert not pw["stays_above_buffer_in_period"]
    rent = next(b for b in pw["bills_made_harder"] if b["description"] == "Shop rent")
    assert rent["due_date"] == "2026-10-05" and rent["amount"] == 12000
    assert rent["status"] == "cannot pay" and rent["status_without_purchase"] == "ok"


def test_financing_says_which_repayments_are_not_checked(ds):
    short = checks.check_spending(ds, 25000, "2026-10-03")
    for o in short["options"]:
        assert o["repayments_inside_forecast"] == 0
        assert o["not_checked"].startswith(f"{o['term_months']} of {o['term_months']} repayments are after 2026-10-31")
    long = checks.check_spending(ds, 25000, "2026-10-03", horizon_days=90)
    loan = next(o for o in long["options"] if o["option_id"] == "F1")
    assert (loan["repayments_inside_forecast"], loan["repayments_after_forecast"]) == (2, 10)
    assert loan["last_repayment"] == "2027-10-03"
    # looks safe over 31 days, but not over 90: the reason the period must be shown
    assert next(o for o in short["options"] if o["option_id"] == "F1")["stays_above_buffer_in_period"]
    assert not loan["stays_above_buffer_in_period"]


def test_small_purchase_is_affordable(ds):
    c = checks.check_spending(ds, 2000, "2026-10-10", "New pots", compare_financing=False)
    assert c["pay_with_cash"]["stays_above_buffer_in_period"]
    assert c["pay_with_cash"]["bills_made_harder"] == []
    assert c["options"] == []


def test_rejects_bad_input(ds):
    with pytest.raises(ValueError):
        checks.check_spending(ds, -5, "2026-10-03")
    with pytest.raises(ValueError, match="inside the forecast period"):
        checks.check_spending(ds, 1000, "2026-12-25")
    with pytest.raises(ValueError):
        checks.reconcile_sales_payments(ds, "2026-09-30", "2026-09-01")
    with pytest.raises(ValueError):
        checks.forecast_cash(ds, 0)


# ---- upload -----------------------------------------------------------------------------
def test_excel_template_round_trip(ds):
    up = build_dataset(read_uploads([("shop.xlsx", to_excel(DEMO))]))
    assert up.dataset_id.startswith("upload-")
    assert checks.closing_balance(up) == checks.closing_balance(ds)
    assert len(checks.reconcile_sales_payments(up)["issues"]) == 4


def test_csv_upload_works():
    files = [(f"{n}.csv", d.to_csv(index=False).encode()) for n, d in read_folder(DEMO).items()]
    assert len(build_dataset(read_uploads(files)).sales) > 0


def test_upload_reports_every_problem_with_rows():
    sheets = pd.read_excel(io.BytesIO(to_excel(DEMO)), sheet_name=None, dtype=str, keep_default_na=False)
    sheets["cash_counts"] = sheets["cash_counts"].rename(columns={"counted_cash": "cash"})
    sheets["sales"].loc[3, "payment_method"] = "card"
    sheets["purchases"].loc[5, "amount_incl_vat"] = ""
    sheets["expenses"].loc[2, "amount"] = "abc"
    del sheets["settings"]
    raw = {k: v for k, v in sheets.items()}
    with pytest.raises(DataError) as e:
        build_dataset(raw)
    text = "\n".join(e.value.problems)
    assert "cash_counts: missing column(s) counted_cash" in text
    assert "payment_method must be cash or qr in row(s) 5" in text
    assert "'amount_incl_vat' is empty in row(s) 7" in text
    assert "'amount' is not a number in row(s) 4" in text
    assert "Missing record type 'settings'" in text


def test_duplicate_ids_are_flagged_not_counted():
    raw = read_folder(DEMO)
    raw["qr_receipts"] = pd.concat([raw["qr_receipts"], raw["qr_receipts"].iloc[[0]]], ignore_index=True)
    with pytest.raises(DataError, match="txn_id' appears more than once"):
        build_dataset(raw)
