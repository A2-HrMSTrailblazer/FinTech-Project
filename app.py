"""BayFlow demo app.  Run:  streamlit run app.py"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from bayflow import checks
from bayflow.agent import Agent, GeminiBackend, ModelUnavailable
from bayflow.data import (DataError, apply_correction_to_dataset, build_dataset, load_dataset, read_uploads,
                          supporting_records, to_excel)
from bayflow.store import Store
from bayflow.tools import Dispatcher, owner_decide

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("BAYFLOW_DATA", ROOT / "data" / "demo"))
DB_PATH = Path(os.getenv("BAYFLOW_DB", ROOT / "data" / "bayflow.db"))

# chart colours: categorical slots 1-2, muted ink for reference lines
C_IN, C_OUT, C_MUTED = "#2a78d6", "#eb6834", "#898781"
STATUS = {"ok": "✅ OK", "below buffer": "⚠️ Below buffer", "cannot pay": "⛔ Cannot pay"}

st.set_page_config(page_title="BayFlow", page_icon="🌊", layout="wide")


@st.cache_resource
def get_store() -> Store:
    return Store(DB_PATH)


store = get_store()
ss = st.session_state
ss.setdefault("session", uuid.uuid4().hex[:8])
ss.setdefault("chat", [])
ss.setdefault("source", "demo")


def baht(x: float) -> str:
    return f"฿{x:,.0f}" if x >= 0 else f"-฿{-x:,.0f}"


def safe_csv(df: pd.DataFrame) -> str:
    """CSV that is safe to open in Excel: text starting with = + - @ is not run as a formula."""
    risky = ("=", "+", "-", "@", "\t", "\r")
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(risky) else v)
    return out.to_csv(index=False)


def mlabel(m: str, fmt: str = "%B %Y") -> str:
    return pd.Period(m, freq="M").strftime(fmt)


def nice_date(d: str) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


def current_dataset():
    if ss.source == "upload" and "upload_raw" in ss:
        ds = build_dataset(ss.upload_raw)
    else:
        ss.source = "demo"
        ds = load_dataset(DATA_DIR)
    for c in store.approved_corrections(ds.dataset_id):
        apply_correction_to_dataset(ds, c)
    return ds


def propose(ds, kind: str, args: dict, reason: str) -> None:
    """Owner-initiated correction: it still goes through the approval queue."""
    from bayflow.data import describe_current_value
    current = describe_current_value(ds, kind, args)
    pid = store.add_proposal(kind, args, current, reason, ss.session, ds.dataset_id)
    store.log("owner", "proposal", "propose_correction", "SUGGEST", {"kind": kind, "args": args}, f"proposal {pid}",
              ss.session)
    st.toast(f"Correction #{pid} added to Approvals.")


try:
    ds = current_dataset()
except DataError as e:
    st.error("Could not load records:\n\n" + "\n".join(f"- {p}" for p in e.problems))
    st.stop()

month = ds.as_of.strftime("%Y-%m")
months = sorted({p.strftime("%Y-%m") for p in pd.period_range(ds.settings["opening_date"], ds.as_of, freq="M")})

# ---- sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("BayFlow")
    st.caption("Cash-flow copilot for small food shops")
    st.write(f"**Records:** {'Demo shop (fictional)' if ss.source == 'demo' else 'Your uploaded records'}")
    st.write(f"**Period:** {ds.settings['opening_date']:%d %b %Y} – {ds.as_of:%d %b %Y}")
    if ss.source == "upload" and st.button("Switch back to demo shop"):
        ss.source = "demo"
        ss.pop("agent", None)
        st.rerun()
    st.divider()
    st.markdown("**What BayFlow may do**")
    st.markdown("- **Read**: check records, explain, forecast\n"
                "- **Suggest**: draft corrections for you\n"
                "- **You approve**: any change to records\n"
                "- **Never**: move money, contact anyone, apply for loans")
    st.divider()
    if st.button("Reset demo (clear corrections)"):
        store.reset_proposals(session=ss.session)
        ss.chat = []
        ss.pop("agent", None)
        st.rerun()

tabs = st.tabs(["Overview", "Cash flow", "Bills & affordability", "Payment check", "Ask BayFlow",
                "Approvals", "Upload records", "Audit log"])
(tab_overview, tab_cash, tab_bills, tab_pay, tab_ask, tab_approve, tab_upload, tab_log) = tabs

recon = checks.reconcile_sales_payments(ds)
summary = checks.cash_flow_summary(ds, month)
cur = summary["month"]
fc31 = checks.forecast_cash(ds, 31)
tax = checks.check_tax_documents(ds, month)

# ---- overview ------------------------------------------------------------------
with tab_overview:
    st.header(f"{mlabel(month, '%B %Y')} at a glance")
    chg = summary.get("change_pct", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash now", baht(cur["closing_cash"]),
              help="Opening cash + money received − money paid, from your records")
    c2.metric("Money received vs sales", baht(recon["money_received"]), delta=baht(recon["difference"]),
              help=f"Recorded sales were {baht(recon['recorded_sales'])}")
    c3.metric("Operating profit", baht(cur["operating_profit"]),
              delta=None if chg.get("operating_profit") is None else f"{chg['operating_profit']:+.1f}% vs last month")
    lp = fc31["lowest_point"]
    c4.metric("Lowest cash, next 31 days", baht(lp["closing_cash"]), help=f"On {nice_date(lp['date'])}")

    st.subheader("Needs your attention")
    alerts = []
    for b in fc31["bills_at_risk"]:
        alerts.append(("Bill", nice_date(b["due_date"]), f"{b['description']}: {baht(b['amount'])}",
                       f"{STATUS[b['status']]}. Cash after paying: {baht(b['cash_after'])}."))
    for i in recon["issues"]:
        if i["type"] != "qr_payment_late":
            alerts.append(("Payment", nice_date(i["date"]), f"{i['type'].replace('_', ' ').capitalize()}: {baht(i['amount'])}",
                           i["explanation"]))
    for c in summary.get("cost_changes", []):
        if c["notable"]:
            alerts.append(("Cost", pd.Period(month, freq="M").strftime("%b %Y"),
                           f"{c['category'].replace('_', ' ').capitalize()} {c['change_pct']:+.0f}%",
                           f"{baht(c['previous'])} → {baht(c['current'])}; now {c['share_of_sales_current_pct']}% of net sales."))
    for p in tax["purchases_missing_tax_invoice"]:
        alerts.append(("Tax papers", nice_date(p["date"]), f"No tax invoice: {p['purchase_id']}",
                       f"{p['supplier']}. {baht(p['vat_amount'])} input VAT cannot be claimed without it."))
    if alerts:
        st.dataframe(pd.DataFrame(alerts, columns=["Area", "Date", "What", "Why it matters"]), hide_index=True,
                     width="stretch")
    else:
        st.success("Nothing needs your attention.")
    if tax["vat"]:
        st.caption(f"VAT for {mlabel(month, '%B')} (estimate {baht(tax['vat']['estimated_vat_payable'])}) "
                   f"is due {nice_date(tax['vat']['due_date'])}. {tax['disclaimer']}")

# ---- cash flow -----------------------------------------------------------------
with tab_cash:
    st.header("Cash flow")
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Month", months[::-1], format_func=lambda m: f"{mlabel(m, '%B %Y')}")
    prev_opts = [m for m in months if m < sel]
    cmp = c2.selectbox("Compare with", prev_opts[::-1] or ["(none)"],
                       format_func=lambda m: m if m == "(none)" else f"{mlabel(m, '%B %Y')}")
    s = checks.cash_flow_summary(ds, sel, None if cmp == "(none)" else cmp)
    m, pm = s["month"], s.get("previous_month")
    if m["partial_month"]:
        st.warning("This month is not complete in the records, so totals are lower than a full month.")
    c = st.columns(4)
    c[0].metric("Opening cash", baht(m["opening_cash"]))
    c[1].metric("Money received", baht(m["money_received"]),
                delta=None if not pm else f"{s['change_pct']['money_received']:+.1f}%")
    c[2].metric("Money paid", baht(m["money_paid"]),
                delta=None if not pm else f"{s['change_pct']['money_paid']:+.1f}%", delta_color="inverse")
    c[3].metric("Closing cash", baht(m["closing_cash"]))
    st.caption("Closing cash = opening cash + money received − money paid")

    led = checks.cash_ledger(ds)
    led = led[led.index.to_period("M").astype(str).isin([x for x in (sel, cmp) if x != "(none)"])]
    weekly = led[["money_received", "money_paid"]].resample("W-SUN").sum().reset_index()
    weekly["week"] = weekly.date.dt.strftime("w/e %d %b")
    long = weekly.melt(id_vars=["date", "week"], value_vars=["money_received", "money_paid"], var_name="flow",
                       value_name="baht")
    long["flow"] = long.flow.map({"money_received": "Money in", "money_paid": "Money out"})
    left, right = st.columns(2)
    with left:
        st.subheader("Money in and out by week")
        bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("week:N", sort=None, title=None),
            xOffset=alt.XOffset("flow:N", sort=["Money in", "Money out"]),
            y=alt.Y("baht:Q", title="Baht", axis=alt.Axis(format=",.0f")),
            color=alt.Color("flow:N", scale=alt.Scale(domain=["Money in", "Money out"], range=[C_IN, C_OUT]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("week:N", title="Week"), alt.Tooltip("flow:N", title=" "),
                     alt.Tooltip("baht:Q", title="Baht", format=",.0f")])
        st.altair_chart(bars.properties(height=300), width="stretch")
    with right:
        st.subheader("Cash balance by day")
        bal = led.reset_index()[["date", "closing_cash"]]
        line = alt.Chart(bal).mark_line(strokeWidth=2, color=C_IN).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("closing_cash:Q", title="Baht", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("closing_cash:Q", title="Cash", format=",.0f")])
        buffer = alt.Chart(pd.DataFrame({"y": [ds.settings["minimum_buffer"]]})).mark_rule(
            strokeDash=[4, 4], color=C_MUTED).encode(y="y:Q")
        st.altair_chart((line + buffer).properties(height=300), width="stretch")
        st.caption(f"Dashed line: your minimum buffer ({baht(ds.settings['minimum_buffer'])}).")

    st.subheader("Profit")
    rows = [("Recorded sales (incl. VAT)", "recorded_sales"), ("Net sales", "net_sales"),
            ("Ingredient cost", "ingredient_cost"), ("Gross profit", "gross_profit")]
    table = [{"Line": label, sel: m[k], **({cmp: pm[k], "Change %": s["change_pct"].get(k)} if pm else {})}
             for label, k in rows]
    for cat in sorted(set(m["operating_expenses"]) | set(pm["operating_expenses"] if pm else [])):
        table.append({"Line": f"  {cat.replace('_', ' ').capitalize()}", sel: m["operating_expenses"].get(cat, 0.0),
                      **({cmp: pm["operating_expenses"].get(cat, 0.0),
                          "Change %": checks._change_pct(m["operating_expenses"].get(cat, 0.0),
                                                         pm["operating_expenses"].get(cat, 0.0))} if pm else {})})
    table.append({"Line": "Operating profit", sel: m["operating_profit"],
                  **({cmp: pm["operating_profit"], "Change %": s["change_pct"]["operating_profit"]} if pm else {})})
    labels = {sel: mlabel(sel, "%b %Y"), **({cmp: mlabel(cmp, "%b %Y")} if pm else {})}
    st.dataframe(pd.DataFrame(table).rename(columns=labels), hide_index=True, width="stretch",
                 column_config={**{v: st.column_config.NumberColumn(format="%,.0f") for v in labels.values()},
                                "Change %": st.column_config.NumberColumn(format="%+.1f%%")})
    st.caption(f"Gross margin {m['gross_margin_pct']}% · operating margin {m['operating_margin_pct']}%. "
               f"Not costs, but cash out: owner drawings {baht(m['owner_drawings'])}, VAT paid {baht(m['vat_paid'])}.")

    st.subheader("Which costs take the biggest share of sales?")
    ratio_rows = [{"Category": k.replace("_", " ").capitalize(), "Month": f"{mlabel(mm, '%b %Y')}",
                   "Share of net sales (%)": v}
                  for mm, src in ((sel, m), (cmp, pm)) if src for k, v in src["expense_to_sales_pct"].items()]
    rdf = pd.DataFrame(ratio_rows)
    order = [f"{mlabel(x, '%b %Y')}" for x in (sel, cmp) if x != "(none)"]
    ratio_chart = alt.Chart(rdf).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("Category:N", sort="-x", title=None), yOffset=alt.YOffset("Month:N", sort=order),
        x=alt.X("Share of net sales (%):Q"),
        color=alt.Color("Month:N", sort=order, scale=alt.Scale(domain=order, range=[C_IN, C_OUT][:len(order)]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Category", "Month", alt.Tooltip("Share of net sales (%):Q", format=".1f")])
    st.altair_chart(ratio_chart.properties(height=320), width="stretch")
    st.caption("Expense-to-sales ratio = expense ÷ net sales × 100")
    if pm:
        notable = [c for c in s["cost_changes"] if c["notable"]]
        for c in notable:
            st.info(f"**{c['category'].replace('_', ' ').capitalize()}** went from {baht(c['previous'])} to "
                    f"{baht(c['current'])} ({c['change_pct']:+.1f}%), and is now {c['share_of_sales_current_pct']}% "
                    f"of net sales (was {c['share_of_sales_previous_pct']}%).")
    with st.expander("Where did the money go? (all money paid, by category)"):
        paid = pd.DataFrame([{"Category": k.replace("_", " ").capitalize(), sel: v,
                              **({cmp: pm["paid_by_category"].get(k, 0.0)} if pm else {})}
                             for k, v in m["paid_by_category"].items()])
        st.dataframe(paid.rename(columns=labels), hide_index=True, width="stretch",
                     column_config={v: st.column_config.NumberColumn(format="%,.0f") for v in labels.values()})
    with st.expander("Formulas used"):
        for k, v in s["formulas"].items():
            st.markdown(f"- **{k.replace('_', ' ')}** = {v}")
        st.caption(s["note"])

# ---- bills & affordability -----------------------------------------------------
with tab_bills:
    st.header("Upcoming bills")
    horizon = st.radio("Forecast period", [31, 60, 90], horizontal=True, format_func=lambda d: f"{d} days")
    fc = checks.forecast_cash(ds, horizon)
    per = fc["forecast_period"]
    st.info(f"**Forecast period: {nice_date(per['start'])} – {nice_date(per['end'])} ({per['days']} days).** "
            f"Starting cash {baht(fc['starting_cash'])}. Nothing after {nice_date(per['end'])} is checked.")
    if fc["bills_at_risk"]:
        st.error(f"{len(fc['bills_at_risk'])} bill(s) would leave you below your {baht(fc['minimum_buffer'])} "
                 f"buffer or unable to pay.")
    bills_df = pd.DataFrame([{"Due": nice_date(b["due_date"]), "Bill": b["description"], "Amount": b["amount"],
                              "Cash after paying": b["cash_after"], "Status": STATUS[b["status"]],
                              "Note": "Can take less" if b["can_be_reduced"] else ""}
                             for b in fc["bill_alerts"]])
    st.dataframe(bills_df, hide_index=True, width="stretch",
                 column_config={"Amount": st.column_config.NumberColumn(format="%,.0f"),
                                "Cash after paying": st.column_config.NumberColumn(format="%,.0f")})
    days = pd.DataFrame(fc["days"])
    days["date"] = pd.to_datetime(days.date)
    fline = alt.Chart(days).mark_line(strokeWidth=2, color=C_IN).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("closing_cash:Q", title="Baht", axis=alt.Axis(format=",.0f")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
                 alt.Tooltip("closing_cash:Q", format=",.0f", title="Cash")])
    fbuf = alt.Chart(pd.DataFrame({"y": [fc["minimum_buffer"]]})).mark_rule(strokeDash=[4, 4], color=C_MUTED).encode(y="y:Q")
    st.altair_chart((fline + fbuf).properties(height=260), width="stretch")
    with st.expander("How the forecast is made"):
        for a in fc["assumptions"]:
            st.markdown(f"- {a}")

    st.header("Can I afford this?")
    with st.form("spend"):
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (฿)", min_value=100, value=25000, step=500)
        when = c2.date_input("Pay on", value=ds.as_of + pd.Timedelta(days=3),
                             min_value=pd.Timestamp(per["start"]), max_value=pd.Timestamp(per["end"]))
        what = c3.text_input("What", value="Second chest freezer")
        go = st.form_submit_button("Check")
    if go:
        r = checks.check_spending(ds, float(amt), when.isoformat(), what, horizon_days=horizon)
        store.log("owner", "tool_call", "check_spending", "READ",
                  {"amount": amt, "date": when.isoformat(), "horizon_days": horizon}, "ok", ss.session)
        st.info(f"**This check covers {nice_date(per['start'])} – {nice_date(per['end'])} ({per['days']} days) only.** "
                "Staying above the buffer in this period does not show that later repayments are affordable.")
        pw = r["pay_with_cash"]
        st.subheader("If you pay cash")
        if pw["stays_above_buffer_in_period"]:
            st.success(f"You stay above your {baht(r['minimum_buffer'])} buffer until {nice_date(per['end'])}. "
                       f"Lowest cash: {baht(pw['lowest_point']['closing_cash'])} on {nice_date(pw['lowest_point']['date'])}.")
        else:
            msg = f"Cash drops below your {baht(r['minimum_buffer'])} buffer on {nice_date(pw['first_day_below_buffer'])}"
            if pw["first_day_negative"]:
                msg += f" and below zero on {nice_date(pw['first_day_negative'])}"
            st.error(msg + ".")
        if pw["bills_made_harder"]:
            st.markdown("**Bills that become hard to pay after this purchase**")
            st.dataframe(pd.DataFrame([{"Due": nice_date(b["due_date"]), "Bill": b["description"], "Amount": b["amount"],
                                        "Cash after paying": b["cash_after"],
                                        "Without the purchase": STATUS[b["status_without_purchase"]],
                                        "With the purchase": STATUS[b["status"]],
                                        "Note": "Can take less" if b["can_be_reduced"] else ""}
                                       for b in pw["bills_made_harder"]]), hide_index=True, width="stretch",
                         column_config={"Amount": st.column_config.NumberColumn(format="%,.0f"),
                                        "Cash after paying": st.column_config.NumberColumn(format="%,.0f")})
        if r["options"]:
            st.subheader("If you use financing")
            st.dataframe(pd.DataFrame([{
                "Option": o["name"], "Monthly payment": o["monthly_payment"], "Months": o["term_months"],
                "Total cost of borrowing": o["total_cost_of_borrowing"],
                "Repayments in forecast": f"{o['repayments_inside_forecast']} of {o['term_months']}",
                "Last repayment": nice_date(o["last_repayment"]),
                "Lowest cash in forecast": o["lowest_point_in_period"]["closing_cash"],
                "Bills made harder": len(o["bills_made_harder"]),
                "Payment vs monthly cash before drawings": o["payment_share_of_monthly_cash_before_drawings_pct"],
            } for o in r["options"]]), hide_index=True, width="stretch",
                column_config={k: st.column_config.NumberColumn(format="%,.0f") for k in
                               ("Monthly payment", "Total cost of borrowing", "Lowest cash in forecast")} |
                {"Payment vs monthly cash before drawings": st.column_config.NumberColumn(format="%.1f%%")})
            for o in r["options"]:
                if o["not_checked"]:
                    st.warning(f"**{o['name']}:** {o['repayments_after_forecast']} of {o['term_months']} repayments "
                               f"fall after {nice_date(per['end'])} and are not checked. Repayments continue until "
                               f"{nice_date(o['last_repayment'])}.")
                if o["bills_made_harder"]:
                    with st.expander(f"{o['name']}: bills made harder"):
                        st.dataframe(pd.DataFrame([{"Due": nice_date(b["due_date"]), "Bill": b["description"],
                                                    "Amount": b["amount"], "Cash after paying": b["cash_after"],
                                                    "Status": STATUS[b["status"]]} for b in o["bills_made_harder"]]),
                                     hide_index=True, width="stretch")
            lt = r["long_term_check"]
            if lt["average_monthly_cash_before_owner_drawings"] is not None:
                st.caption(f"Long-term guide: in an average full month ({', '.join(lt['based_on_months'])}) the business "
                           f"has {baht(lt['average_monthly_cash_before_owner_drawings'])} left before the owner takes money "
                           f"out. {lt['explanation']}")
            st.caption(r["financing_note"])

# ---- payment check -------------------------------------------------------------
with tab_pay:
    st.header("Did the money arrive?")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=pd.Timestamp(recon["period"]["start"]),
                          min_value=ds.settings["opening_date"], max_value=ds.as_of)
    end = c2.date_input("To", value=ds.as_of, min_value=ds.settings["opening_date"], max_value=ds.as_of)
    rc = checks.reconcile_sales_payments(ds, start.isoformat(), end.isoformat())
    c = st.columns(4)
    c[0].metric("Recorded sales", baht(rc["recorded_sales"]))
    c[1].metric("QR / bank received", baht(rc["qr_received"]))
    c[2].metric("Cash counted", baht(rc["cash_counted"]))
    c[3].metric("Difference", baht(rc["difference"]))
    st.caption(rc["note"])
    if not rc["issues"]:
        st.success("Every sale in this period matches money received.")
    for n, i in enumerate(rc["issues"]):
        with st.expander(f"{nice_date(i['date'])} · {i['type'].replace('_', ' ').capitalize()} · {baht(i['amount'])}"):
            st.write(i["explanation"])
            st.write(f"**Suggested:** {i['suggested_action']}")
            for name, rows in supporting_records(ds, i["evidence"], i["date"]).items():
                st.markdown(f"*{name.replace('_', ' ').capitalize()}*")
                st.dataframe(rows, hide_index=True, width="stretch")
            if i["type"] == "possible_double_entry":
                if st.button(f"Suggest removing bill {i['evidence']['bill_id']}", key=f"void{n}"):
                    propose(ds, "void_bill", {"bill_id": i["evidence"]["bill_id"]},
                            f"Identical to {i['evidence']['identical_to_bill']}; keyed in twice")
            if i["type"] in ("cash_short", "cash_over"):
                new = st.number_input("Correct cash count (if the count was wrong)", min_value=0.0,
                                      value=float(i["evidence"]["cash_counted"]), key=f"cc{n}")
                if st.button("Suggest this correction", key=f"ccb{n}"):
                    propose(ds, "set_cash_count", {"date": i["date"], "counted_cash": new}, "Recount by owner")

    st.header("Tax papers")
    if tax["purchases_missing_tax_invoice"]:
        for p in tax["purchases_missing_tax_invoice"]:
            c1, c2 = st.columns([4, 1])
            c1.write(f"{nice_date(p['date'])} · {p['supplier']} · {p['purchase_id']} · {baht(p['amount_incl_vat'])} "
                     f"(VAT {baht(p['vat_amount'])})")
            if c2.button("I have the invoice", key=f"inv{p['purchase_id']}"):
                propose(ds, "set_tax_invoice", {"purchase_id": p["purchase_id"], "has_tax_invoice": "yes"},
                        "Owner found the tax invoice")
    else:
        st.success("All purchases from VAT-registered suppliers have a tax invoice.")
    st.caption(tax["disclaimer"])

# ---- ask -----------------------------------------------------------------------
with tab_ask:
    st.header("Ask BayFlow")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("Add GEMINI_API_KEY to a .env file to use the assistant. The other tabs work without it.")
    else:
        agent_key = (ds.dataset_id, len(store.approved_corrections(ds.dataset_id)))
        if ss.get("agent_key") != agent_key:
            ss.agent = Agent(GeminiBackend(), Dispatcher(ds, store, session=ss.session), store)
            ss.agent_key = agent_key
        st.caption("Try: *Did I receive all the money from last month's sales?* · *Why did my cash go down?* · "
                   "*Which costs take the biggest share of my sales?* · *Can I buy a 25,000 baht freezer on 3 October?*")
        for msg in ss.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                if msg.get("tools"):
                    st.caption("Checked with: " + ", ".join(t["tool"] for t in msg["tools"]))
                if msg.get("unverified"):
                    st.warning("These numbers could not be traced to your records, please double-check: "
                               + ", ".join(msg["unverified"]))
        if ss.get("ask_error"):
            st.warning(ss.pop("ask_error"))
        q = st.chat_input("Ask about your sales, cash, bills or tax papers")
        if q:
            ss.chat.append({"role": "user", "text": q})
            with st.spinner("Checking your records…"):
                try:
                    r = ss.agent.ask(q)
                    ss.chat.append({"role": "assistant", "text": r.text, "tools": r.tool_calls,
                                    "unverified": r.unverified_numbers})
                except ModelUnavailable as e:
                    ss.chat.pop()
                    ss.ask_error = f"{e} The other tabs still work without the assistant."
                except Exception as e:
                    ss.chat.append({"role": "assistant", "text": f"Sorry, the assistant failed: {e}"})
            st.rerun()

# ---- approvals -----------------------------------------------------------------
with tab_approve:
    st.header("Corrections waiting for you")
    st.caption("Nothing in your records changes until you approve. Your original files are never edited.")
    pending = store.proposals("pending", ds.dataset_id)
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
                st.rerun()
            if c2.button("Reject", key=f"r{p['id']}"):
                owner_decide(store, p["id"], False)
                st.rerun()
    done = [p for p in store.proposals(dataset=ds.dataset_id) if p["status"] != "pending"]
    if done:
        st.subheader("Decided")
        st.dataframe(pd.DataFrame(done)[["id", "kind", "args", "status", "decided_ts"]], hide_index=True,
                     width="stretch")

# ---- upload --------------------------------------------------------------------
with tab_upload:
    st.header("Use your own records")
    st.markdown("Upload **one Excel file** with one sheet per record type, or **several CSV files** named after "
                "the record types. Download the template to see the exact columns, with the demo shop's records "
                "filled in as an example.")
    st.download_button("Download Excel template", to_excel(DATA_DIR), "bayflow_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    files = st.file_uploader("Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)
    if files and st.button("Check and use these records"):
        try:
            raw = read_uploads([(f.name, f.getvalue()) for f in files])
            new = build_dataset(raw)
        except DataError as e:
            store.log("owner", "upload", outcome="rejected: " + "; ".join(e.problems)[:500], session=ss.session)
            st.error("Please fix these in your file and upload again:\n\n" + "\n".join(f"- {p}" for p in e.problems))
        else:
            ss.upload_raw, ss.source = raw, "upload"
            store.log("owner", "upload", args={"files": [f.name for f in files]}, outcome=new.dataset_id,
                      session=ss.session)
            st.rerun()
    if ss.source == "upload":
        st.success(f"Using your uploaded records ({ds.settings['opening_date']:%d %b %Y} – {ds.as_of:%d %b %Y}).")
        st.dataframe(pd.DataFrame([{"Record type": k, "Rows": len(v)} for k, v in ds.tables.items()]),
                     hide_index=True)
        for w in ds.warnings:
            st.warning(w)

# ---- audit log -----------------------------------------------------------------
with tab_log:
    st.header("Audit log")
    st.caption("Every check, refusal, proposal, decision and upload. Export this as execution evidence.")
    log = pd.DataFrame(store.audit_rows())
    if log.empty:
        st.write("No activity yet.")
    else:
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("Download CSV", safe_csv(log), "bayflow_audit_log.csv", "text/csv")
