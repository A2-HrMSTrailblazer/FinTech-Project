# BayFlow — SME Finance Copilot

BayFlow is an AI finance copilot for small food shops. It answers three questions
from the records an owner already keeps:

1. **Did we receive the money recorded as sales?** Matches QR bills to PromptPay
   receipts and cash sales to the daily cash count.
2. **Where is money leaking, and why is cash going up or down?** Compares the
   ingredients used with what sales need, and spending month by month.
3. **How much can we safely spend?** Forecasts cash with upcoming bills and the
   VAT payment, and compares paying cash with financing options.

Entry for the 2026 FinTechathon International Track, Topic E (SME Finance Copilot).

## Design in one paragraph

All numbers come from plain, tested Python functions (`bayflow/checks.py`). The
language model (Gemini) only decides which check to run and explains the result.
Every finding names the record IDs it came from, and every number in an AI
answer is automatically checked against the tool results; numbers that cannot
be traced are flagged to the owner. The agent can read and suggest, but only
the owner can approve a change, and no tool exists that moves money.

## Permission tiers

| Tier | What | Who |
|---|---|---|
| READ | Run checks, explain, forecast | Agent |
| SUGGEST | Create a pending correction (`propose_correction`) | Agent |
| APPROVE | Apply a correction to the records | Owner only (UI) |
| NEVER | Move money, contact anyone, apply for credit | No such tool exists |

Tiers are enforced in `bayflow/tools.py`, not in the prompt. Refused calls are
written to the audit log. Original files are never edited: approved corrections
are stored separately and applied when the records are loaded.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your GEMINI_API_KEY in .env
python scripts/generate_demo_data.py   # optional: rebuild the demo data
pytest                       # 25 tests, run offline, no API key needed
streamlit run app.py
```

Every tab except **Ask BayFlow** works without an API key.

The assistant uses `GEMINI_MODEL` (default `gemini-flash-latest`). If Gemini is busy (429/5xx), BayFlow retries twice and then tries each model in `GEMINI_FALLBACK_MODELS`. If all are busy, the question is undone so the owner can simply ask again.

## Project layout

```
app.py                     Streamlit demo app
bayflow/checks.py          Deterministic checks (all numbers come from here)
bayflow/tools.py           Tool registry, permission tiers, dispatcher
bayflow/agent.py           Gemini tool-calling loop and number self-check
bayflow/data.py            Record schema, loading, corrections
bayflow/store.py           SQLite audit log and approval queue
config/tax_rules.json      VAT rate and PP.30 deadlines (verify at rd.go.th)
data/demo/                 Fictional shop records, Aug–Sep 2026
scripts/generate_demo_data.py  Builds the demo data and its answer key
tests/                     Checks vs. answer key, permissions, agent loop
```

## Demo data

`data/demo/` is a fictional fresh-food shop (about ฿160,000 sales a month,
VAT registered). September has planted problems, listed in `answer_key.json`:

- a QR payment that never arrived, and one that arrived late (not missing)
- a cash bill keyed in twice, and a cash drawer short by ฿1,000
- pork used but not billed on three days, and one wrong egg count
- two supplier purchases without a tax invoice
- an instruction hidden in a purchase note (prompt-injection test)
- electricity up 46% from August

The tests check that BayFlow finds every planted problem and reports nothing in
the clean month (August).

## Record format

| File | Columns |
|---|---|
| sales.csv | bill_id, timestamp, item_id, qty, unit_price, line_total, payment_method (cash/qr) |
| qr_receipts.csv | txn_id, timestamp, amount, reference |
| cash_counts.csv | date, counted_cash |
| purchases.csv | purchase_id, date, supplier, ingredient_id, qty, amount_incl_vat, vat_amount, has_tax_invoice, note |
| stock_counts.csv | date, ingredient_id, closing_count |
| expenses.csv | expense_id, date, category, description, amount |
| upcoming_bills.csv | due_date, category, description, amount |
| menu.csv, recipes.csv, ingredients.csv | menu, ingredient use per item, unit cost and waste allowance |
| cash_position.json | as_of, bank_balance, cash_on_hand, minimum_buffer |

## Limits

- Tax figures are planning estimates. The owner's accountant files returns.
- Financing products in the demo are fictional.
- Stock checks cover the key ingredients the owner counts, not every item.
- Forecasts use the average of the last 28 days by weekday.
