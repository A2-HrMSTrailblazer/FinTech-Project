"""BayFlow demo app.  Run:  streamlit run app.py"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from bayflow import checks
from bayflow.agent import Agent, GeminiBackend
from bayflow.data import DataError, load_dataset
from bayflow.store import Store
from bayflow.tools import Dispatcher, owner_decide

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("BAYFLOW_DATA", ROOT / "data" / "demo"))
DB_PATH = Path(os.getenv("BAYFLOW_DB", ROOT / "data" / "bayflow.db"))

st.set_page_config(page_title="BayFlow", page_icon="🌊", layout="wide")


@st.cache_resource
def get_store() -> Store:
    return Store(DB_PATH)


store = get_store()
if "session" not in st.session_state:
    st.session_state.session = uuid.uuid4().hex[:8]
    st.session_state.chat = []


def current_dataset():
    corrections = store.approved_corrections()
    try:
        return load_dataset(DATA_DIR, corrections)
    except DataError as e:
        st.error(f"Could not load records: {e}")
        st.stop()


ds = current_dataset()
baht = lambda x: f"฿{x:,.0f}"

# ---- sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("BayFlow")
    st.caption("Finance copilot for small food shops")
    st.write(f"**Records as of:** {ds.cash_position['as_of']}")
    st.write(f"**Data folder:** `{DATA_DIR.relative_to(ROOT) if DATA_DIR.is_relative_to(ROOT) else DATA_DIR}`")
    st.divider()
    st.markdown("**What BayFlow may do**")
    st.markdown("- **Read**: check records, explain, forecast\n"
                "- **Suggest**: draft corrections for you\n"
                "- **You approve**: any change to records\n"
                "- **Never**: move money, contact anyone, apply for loans")
    st.divider()
    if st.button("Reset demo (clear approvals and log)"):
        store.db.executescript("DELETE FROM proposals; DELETE FROM audit_log;")
        store.db.commit()
        st.session_state.chat = []
        st.session_state.pop("agent", None)
        st.rerun()

tab_overview, tab_ask, tab_records, tab_cash, tab_approve, tab_log = st.tabs(
    ["Overview", "Ask BayFlow", "Record checks", "Cash & spending", "Approvals", "Audit log"])

month = pd.Timestamp(ds.cash_position["as_of"]).strftime("%Y-%m")
prev_month = (pd.Period(month, freq="M") - 1).strftime("%Y-%m")
recon = checks.reconcile_sales_payments(ds)
stock = checks.stock_variance(ds)
tax = checks.check_tax_documents(ds, month)
fc = checks.forecast_cash(ds, 31)
change = checks.explain_cash_change(ds, prev_month, month)

# ---- overview ------------------------------------------------------------------
with tab_overview:
    st.header(f"{pd.Period(month, freq='M').strftime('%B %Y')} at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded sales", baht(recon["recorded_sales"]))
    c2.metric("Money received", baht(recon["money_received"]), delta=baht(recon["difference"]),
              delta_color="normal" if recon["difference"] >= 0 else "inverse")
    c3.metric("Unexplained stock usage", baht(stock["unexplained_usage_value_thb"]))
    c4.metric("Input VAT at risk", baht(tax["input_vat_at_risk"]))
    st.subheader("Needs your attention")
    items = []
    for i in recon["issues"]:
        if i["type"] != "qr_payment_late":
            items.append((i["date"], i["type"].replace("_", " "), baht(i["amount"]), i["explanation"]))
    for f in stock["findings"]:
        items.append((f["date"], f"{f['type'].replace('_', ' ')}: {f['ingredient']}",
                      f"{f['difference']} {f['unit']}", f["explanation"]))
    for p in tax["purchases_missing_tax_invoice"]:
        items.append((p["date"], "missing tax invoice", baht(p["vat_amount"]) + " VAT", f"{p['supplier']} ({p['purchase_id']})"))
    st.dataframe(pd.DataFrame(items, columns=["Date", "What", "Amount", "Why it matters"]).sort_values("Date"),
                 hide_index=True, width="stretch")
    lp = fc["lowest_point"]
    st.info(f"Cash today: **{baht(fc['starting_cash'])}**. Lowest point in the next 31 days: "
            f"**{baht(lp['closing_balance'])}** on {lp['date']}. VAT for {month} (estimate "
            f"{baht(tax['vat']['estimated_vat_payable'])}) is due {tax['vat']['due_date']}.")

# ---- ask -----------------------------------------------------------------------
with tab_ask:
    st.header("Ask BayFlow")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("Add GEMINI_API_KEY to a .env file to use the assistant. The other tabs work without it.")
    else:
        if "agent" not in st.session_state:
            disp = Dispatcher(ds, store, session=st.session_state.session)
            st.session_state.agent = Agent(GeminiBackend(), disp, store)
        st.caption("Try: *Did I receive all the money from last month's sales?* · *Why did my cash go down?* · "
                   "*Can I buy a 25,000 baht freezer on 3 October?* · *ขายได้เงินครบไหมเดือนที่แล้ว*")
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])
                if m.get("tools"):
                    st.caption("Checked with: " + ", ".join(t["tool"] for t in m["tools"]))
                if m.get("unverified"):
                    st.warning("These numbers could not be traced to your records, please double-check: "
                               + ", ".join(m["unverified"]))
        q = st.chat_input("Ask about your sales, stock, cash or tax papers")
        if q:
            st.session_state.chat.append({"role": "user", "text": q})
            with st.spinner("Checking your records…"):
                try:
                    r = st.session_state.agent.ask(q)
                    st.session_state.chat.append({"role": "assistant", "text": r.text, "tools": r.tool_calls,
                                                  "unverified": r.unverified_numbers})
                except Exception as e:  # network/API errors are shown, not hidden
                    st.session_state.chat.append({"role": "assistant", "text": f"Sorry, the assistant failed: {e}"})
            st.rerun()

# ---- record checks -------------------------------------------------------------
with tab_records:
    st.header("Did the money arrive?")
    st.caption(recon["note"])
    st.dataframe(pd.DataFrame([{"Date": i["date"], "Type": i["type"].replace("_", " "), "Amount": i["amount"],
                                "Records": ", ".join(f"{k}: {v}" for k, v in i["evidence"].items()),
                                "Suggested action": i["suggested_action"]} for i in recon["issues"]]),
                 hide_index=True, width="stretch")
    st.header("Stock: expected vs actual usage")
    st.caption(stock["method"])
    st.dataframe(pd.DataFrame([{"Date": f["date"], "Ingredient": f["ingredient"], "Type": f["type"].replace("_", " "),
                                "Expected": f["expected_usage"], "Actual": f["actual_usage"], "Unit": f["unit"],
                                "Value (฿)": f["value_thb"], "Explanation": f["explanation"]}
                               for f in stock["findings"]]), hide_index=True, width="stretch")
    st.header("Tax papers")
    st.dataframe(pd.DataFrame(tax["purchases_missing_tax_invoice"]), hide_index=True, width="stretch")
    st.caption(tax["disclaimer"])

# ---- cash ----------------------------------------------------------------------
with tab_cash:
    st.header("Why did cash change?")
    st.dataframe(pd.DataFrame(change["spending_changes"]), hide_index=True, width="stretch")
    st.caption(change["note"])
    st.header("Next 31 days")
    days = pd.DataFrame(fc["days"]).set_index("date")
    st.line_chart(days["closing_balance"])
    with st.expander("Assumptions"):
        for a in fc["assumptions"]:
            st.markdown(f"- {a}")
    st.header("Can I afford it?")
    with st.form("spend"):
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (฿)", min_value=100, value=25000, step=500)
        when = c2.date_input("When", value=pd.Timestamp(ds.cash_position["as_of"]) + pd.Timedelta(days=3))
        what = c3.text_input("What", value="Second chest freezer")
        go = st.form_submit_button("Check")
    if go:
        r = checks.check_spending(ds, float(amt), when.isoformat(), what)
        store.log("owner", "tool_call", "check_spending", "READ", {"amount": amt, "date": when.isoformat()}, "ok",
                  st.session_state.session)
        pw = r["pay_with_cash"]
        if pw["stays_above_buffer"]:
            st.success(f"Paying cash keeps you above your {baht(r['minimum_buffer'])} buffer. "
                       f"Lowest point: {baht(pw['lowest_point']['closing_balance'])} on {pw['lowest_point']['date']}.")
        else:
            msg = f"Paying cash drops you below your {baht(r['minimum_buffer'])} buffer on {pw['first_day_below_buffer']}"
            if pw["first_day_negative"]:
                msg += f", and below zero on {pw['first_day_negative']}"
            st.error(msg + ".")
        st.dataframe(pd.DataFrame([{"Option": o["name"], "Monthly payment": o["monthly_payment"],
                                    "Months": o["term_months"], "Total cost of borrowing": o["total_cost_of_borrowing"],
                                    "Lowest cash": o["lowest_point"]["closing_balance"],
                                    "Stays above buffer": o["stays_above_buffer"]} for o in r["options"]]),
                     hide_index=True, width="stretch")
        st.caption(r.get("financing_note", ""))

# ---- approvals -----------------------------------------------------------------
with tab_approve:
    st.header("Corrections waiting for you")
    st.caption("BayFlow can only suggest. Nothing in your records changes until you approve.")
    pending = store.proposals("pending")
    if not pending:
        st.write("Nothing waiting.")
    for p in pending:
        with st.container(border=True):
            st.markdown(f"**#{p['id']} · {p['kind'].replace('_', ' ')}** · {p['args']}")
            st.write(f"Current value: {p['current_value']}")
            st.write(f"Reason: {p['reason']}")
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"a{p['id']}"):
                owner_decide(store, p["id"], True)
                st.session_state.pop("agent", None)   # agent reloads the corrected records
                st.rerun()
            if c2.button("Reject", key=f"r{p['id']}"):
                owner_decide(store, p["id"], False)
                st.rerun()
    done = [p for p in store.proposals() if p["status"] != "pending"]
    if done:
        st.subheader("Decided")
        st.dataframe(pd.DataFrame(done)[["id", "kind", "args", "status", "decided_ts"]], hide_index=True,
                     width="stretch")

# ---- audit log -----------------------------------------------------------------
with tab_log:
    st.header("Audit log")
    st.caption("Every tool call, refusal, proposal and decision. Export this as execution evidence.")
    log = pd.DataFrame(store.audit_rows())
    if log.empty:
        st.write("No activity yet.")
    else:
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("Download CSV", log.to_csv(index=False), "bayflow_audit_log.csv", "text/csv")
