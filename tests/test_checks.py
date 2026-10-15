"""The checks must find every planted problem in the demo data, and nothing else.

The answer key is written by the data generator, not by the checks, so these
tests compare BayFlow against an independent source of truth.
"""
import json
from pathlib import Path

import pytest

from bayflow import checks
from bayflow.data import DataError, load_dataset

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


def _issues(recon, kind):
    return [i for i in recon["issues"] if i["type"] == kind]


def test_finds_missed_qr_payment(recon, key):
    found = _issues(recon, "qr_payment_not_found")
    assert [i["evidence"]["bill_id"] for i in found] == [k["bill_id"] for k in key["missed_qr"]]
    assert found[0]["amount"] == key["missed_qr"][0]["amount"]


def test_late_payment_is_not_reported_as_missing(recon, key):
    late = _issues(recon, "qr_payment_late")
    assert [i["evidence"]["bill_id"] for i in late] == [k["bill_id"] for k in key["late_qr"]]


def test_double_entry_explains_cash_gap(recon, key):
    dup = _issues(recon, "possible_double_entry")
    assert len(dup) == 1
    assert dup[0]["evidence"]["bill_id"] == key["duplicate_bills"][0]["duplicate"]
    assert dup[0]["evidence"]["identical_to_bill"] == key["duplicate_bills"][0]["original"]


def test_cash_shortfall(recon, key):
    short = _issues(recon, "cash_short")
    assert [(i["date"], i["amount"]) for i in short] == [(k["date"], k["amount"]) for k in key["cash_short"]]


def test_no_other_payment_issues(recon):
    allowed = {"qr_payment_not_found", "qr_payment_late", "possible_double_entry", "cash_short"}
    assert {i["type"] for i in recon["issues"]} == allowed
    assert len(recon["issues"]) == 4


def test_totals_add_up(recon):
    assert recon["difference"] == pytest.approx(recon["money_received"] - recon["recorded_sales"])
    # every baht of difference is explained by an issue (late payment arrived next day, inside the month)
    assert -recon["difference"] == pytest.approx(recon["amount_needing_check"])


def test_clean_month_has_no_issues(ds):
    assert checks.reconcile_sales_payments(ds, "2026-08-01", "2026-08-31")["issues"] == []
    assert checks.stock_variance(ds, "2026-08-01", "2026-08-31")["findings"] == []


def test_stock_finds_unbilled_usage_and_miscount(ds, key):
    s = checks.stock_variance(ds, "2026-09-01", "2026-09-30")
    unexplained = [(f["date"], f["ingredient_id"]) for f in s["findings"] if f["type"] == "unexplained_usage"]
    assert unexplained == [(k["date"], k["ingredient_id"]) for k in key["unbilled_usage"]]
    count_errors = [(f["date"], f["ingredient_id"]) for f in s["findings"] if f["type"] == "likely_count_error"]
    assert count_errors == [(k["date"], k["ingredient_id"]) for k in key["miscount"]]
    assert len(s["findings"]) == 4
    assert s["unexplained_usage_value_thb"] > 0


def test_explains_electricity_increase(ds, key):
    e = checks.explain_cash_change(ds, "2026-08", "2026-09")
    elec = next(c for c in e["spending_changes"] if c["category"] == "electricity")
    assert elec["2026-08"] == key["expense_change"]["august"]
    assert elec["2026-09"] == key["expense_change"]["september"]
    assert elec["notable"]


def test_missing_tax_invoices(ds, key):
    t = checks.check_tax_documents(ds, "2026-09")
    assert sorted(p["purchase_id"] for p in t["purchases_missing_tax_invoice"]) == \
        sorted(k["purchase_id"] for k in key["missing_tax_invoice"])
    assert t["input_vat_at_risk"] == pytest.approx(sum(k["vat_amount"] for k in key["missing_tax_invoice"]))
    assert t["vat"]["due_date"] == "2026-10-23"


def test_pp30_falls_back_to_paper_deadline_after_extension():
    assert checks.pp30_due_date("2026-12").isoformat() == "2027-01-23"
    assert checks.pp30_due_date("2027-01").isoformat() == "2027-02-15"


def test_forecast_and_spending_decision(ds):
    base = checks.forecast_cash(ds, 31)
    assert base["first_day_negative"] is None
    vat_row = [d for d in base["days"] if any("VAT" in b["description"] for b in d["bills"])]
    assert vat_row and vat_row[0]["date"] == "2026-10-23"

    c = checks.check_spending(ds, 25000, "2026-10-03", "Second chest freezer")
    assert c["pay_with_cash"]["first_day_negative"] == "2026-10-05"   # rent day
    assert not c["pay_with_cash"]["stays_above_buffer"]
    assert all("FICTIONAL" in o["name"] for o in c["options"])


def test_small_purchase_is_affordable(ds):
    c = checks.check_spending(ds, 2000, "2026-10-10", "New pots", compare_financing=False)
    assert c["pay_with_cash"]["stays_above_buffer"]
    assert c["options"] == []


def test_rejects_bad_input(ds):
    with pytest.raises(ValueError):
        checks.check_spending(ds, -5, "2026-10-03")
    with pytest.raises(ValueError):
        checks.stock_variance(ds, ingredient_id="ING-NOPE")
    with pytest.raises(ValueError):
        checks.reconcile_sales_payments(ds, "2026-09-30", "2026-09-01")


def test_missing_column_is_reported(tmp_path):
    import shutil
    for f in DEMO.iterdir():
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "cash_counts.csv").write_text("date,amount\n2026-09-01,10\n")
    with pytest.raises(DataError, match="counted_cash"):
        load_dataset(tmp_path)
