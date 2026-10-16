# BayFlow — SME Finance Copilot

BayFlow is an AI cash-flow copilot for small food shops. It works from the
records an owner already keeps in Excel and answers three questions:

1. **Did we receive the money recorded as sales?** Matches QR bills to QR/bank
   receipts and cash sales to the daily cash count, and flags differences.
2. **Where did the money go, and why did cash change?** Cash in and out, profit,
   and which costs take a larger share of sales than last month.
3. **Can we pay what is coming, and can we afford this?** Upcoming bills with
   the cash left after each one, and how a planned purchase changes that.

Entry for the 2026 FinTechathon International Track, Topic E (SME Finance Copilot).

## Features

| Feature | Where |
|---|---|
| Excel/CSV upload with clear error messages and a template | *Upload records* tab |
| Sales and payment matching, with supporting records for each finding | *Payment check* tab |
| Cash-flow dashboard: cash in/out, profit, expense-to-sales ratios, month comparison | *Cash flow* tab |
| Upcoming bill alerts (OK / below buffer / cannot pay) | *Bills & affordability* tab, *Overview* |
| "Can I afford this?" with the bills that become hard to pay and the forecast period stated | *Bills & affordability* tab |
| AI explanations (Gemini) that only use the checked numbers | *Ask BayFlow* tab |
| Review and approval of corrections; audit log | *Approvals*, *Audit log* tabs |

Stock and ingredient counting is out of scope: the interviewed owner does not
count ingredients daily, so BayFlow uses records they already have.

## Formulas

| Measure | Formula |
|---|---|
| Closing cash | opening cash + money received − money paid |
| Net sales | recorded sales × 100 / 107 (VAT-registered shop) |
| Ingredient cost | ingredient purchases − input VAT claimable with a tax invoice |
| Gross profit | net sales − ingredient cost |
| Operating profit | gross profit − operating expenses (owner drawings and VAT payments are not costs) |
| Expense-to-sales ratio | expense ÷ net sales × 100 |
| Period change | (current − previous) ÷ previous × 100 |
| Bill status | cash after paying the bill: below 0 = cannot pay; below the owner's buffer = below buffer |

Profit is not cash: cash also falls when the owner takes money out or pays VAT.

## Design

All numbers come from plain, tested Python functions (`bayflow/checks.py`). The
language model only decides which check to run and explains the result. Every
finding names the record IDs it came from, and every number in an AI answer is
checked against the tool results; numbers that cannot be traced are flagged.

**Forecast period.** Forecasts cover 31, 60 or 90 days and always state the
period. For financing, BayFlow lists how many repayments fall after the period
and are not checked, so "stays above the buffer for 31 days" is never read as
"affordable for the whole 6–12 month plan".

## Permission tiers

| Tier | What | Who |
|---|---|---|
| READ | Run checks, explain, forecast | Agent |
| SUGGEST | Create a pending correction | Agent or owner |
| APPROVE | Apply a correction to the records | Owner only (UI) |
| NEVER | Move money, contact anyone, apply for credit | No such tool exists |

Tiers are enforced in `bayflow/tools.py`, not in the prompt. Refused calls are
written to the audit log. Uploaded files are never edited: approved corrections
are stored separately for each set of records and applied when they are loaded.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your GEMINI_API_KEY in .env
python scripts/generate_demo_data.py   # optional: rebuild the demo data
pytest                       # runs offline, no API key needed
streamlit run app.py
```

Every tab except **Ask BayFlow** works without an API key. The assistant uses
`GEMINI_MODEL` (default `gemini-flash-latest`). If Gemini is busy (429/5xx),
BayFlow retries twice and then tries each model in `GEMINI_FALLBACK_MODELS`.

## Record format

Upload one Excel file with one sheet per record type, or one CSV file per record
type. The *Upload records* tab has a template with the demo shop filled in.

| Sheet / file | Required columns | Optional columns |
|---|---|---|
| sales | bill_id, timestamp, line_total, payment_method (cash/qr) | item, qty, unit_price |
| qr_receipts | txn_id, timestamp, amount | reference |
| cash_counts | date, counted_cash | |
| purchases | purchase_id, date, supplier, amount_incl_vat | item, qty, unit, vat_amount, has_tax_invoice (yes/no), note |
| expenses | date, category, amount | expense_id, description |
| upcoming_bills | due_date, description, amount | category, repeat (none/weekly/monthly) |
| settings | opening_date, opening_balance, minimum_buffer | as_of, vat_registered (yes/no) |
| financing_options | option_id, name, annual_rate_pct, fee_thb, term_months | (whole sheet optional) |

Use the expense category `owner_drawings` for money the owner takes out and
`vat_payment` for VAT paid. All problems in a file are reported at once with row
numbers; BayFlow asks for missing values instead of guessing them.

## Demo data

`data/demo/` is a fictional fresh-food shop (about ฿160,000 sales a month,
VAT registered), August–September 2026. September has planted problems, listed
in `answer_key.json`:

- a QR payment that never arrived, and one that arrived late (not missing)
- a cash bill keyed in twice, and a cash drawer short by ฿1,000
- two supplier purchases without a tax invoice
- an instruction hidden in a purchase note (prompt-injection test)
- electricity up 46% and packaging up 33% from August

The tests check that BayFlow finds every planted problem, reports nothing in the
clean month (August), and that the books close at the balance in the answer key.

## Project layout

```
app.py                     Streamlit app
bayflow/checks.py          Calculations (all numbers come from here)
bayflow/tools.py           Tool registry, permission tiers, dispatcher
bayflow/agent.py           Gemini tool-calling loop and number self-check
bayflow/data.py            Record format, upload validation, corrections
bayflow/store.py           SQLite audit log and approval queue
config/tax_rules.json      VAT rate and PP.30 deadlines (verify at rd.go.th)
data/demo/                 Fictional shop records and answer key
scripts/generate_demo_data.py  Builds the demo data
tests/                     Checks vs. answer key, hand-worked examples, permissions, agent
```

## Limits

- Tax figures are planning estimates. The owner's accountant files returns.
- Financing products in the demo are fictional.
- Forecasts use the average of the last 28 days by weekday; nothing after the
  forecast period is checked.
- Ingredient purchases are treated as used in the month they are bought.
