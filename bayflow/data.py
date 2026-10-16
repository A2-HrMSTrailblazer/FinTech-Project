"""Load and validate the owner's records.

Records can come from a folder of CSV files, from uploaded CSV files, or from
one Excel workbook with one sheet per record type. Sheet and file names are the
record names below (e.g. "sales", "qr_receipts"). Problems are collected and
reported all at once, with row numbers, so the owner can fix them in their own
file. BayFlow asks for missing information; it never guesses it.

Original files are never modified: approved corrections are stored separately
and applied on top when the records are loaded.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# name: (required columns, optional columns with default values, description)
SCHEMA: dict[str, tuple[list[str], dict, str]] = {
    "sales": (["bill_id", "timestamp", "line_total", "payment_method"],
              {"item": "", "qty": 1, "unit_price": None},
              "One row per item sold. payment_method is cash or qr."),
    "qr_receipts": (["txn_id", "timestamp", "amount"], {"reference": ""},
                    "Money received by PromptPay/QR or bank transfer (from the bank app export)."),
    "cash_counts": (["date", "counted_cash"], {},
                    "Cash counted in the drawer at the end of each day."),
    "purchases": (["purchase_id", "date", "supplier", "amount_incl_vat"],
                  {"item": "", "qty": None, "unit": "", "vat_amount": 0.0, "has_tax_invoice": "no", "note": ""},
                  "Ingredients and supplies bought, paid on the date shown."),
    "expenses": (["date", "category", "amount"], {"expense_id": None, "description": ""},
                 "Other money paid out: rent, wages, bills, VAT payments, owner drawings."),
    "upcoming_bills": (["due_date", "description", "amount"], {"category": "", "repeat": "none"},
                       "Bills still to pay. repeat is none, weekly or monthly."),
    "settings": (["opening_date", "opening_balance", "minimum_buffer"], {"as_of": None, "vat_registered": "yes"},
                 "One row: cash (bank + drawer) on the morning of opening_date, the minimum cash to keep, and whether the shop is VAT registered."),
    "financing_options": (["option_id", "name", "annual_rate_pct", "fee_thb", "term_months"], {},
                          "Optional. Financing products to compare."),
}
OPTIONAL_TABLES = {"financing_options"}
DATE_COLS = {"date", "due_date", "opening_date", "as_of"}
TIME_COLS = {"timestamp"}
NUM_COLS = {"line_total", "qty", "unit_price", "amount", "counted_cash", "amount_incl_vat", "vat_amount",
            "opening_balance", "minimum_buffer", "annual_rate_pct", "fee_thb", "term_months"}
ID_COLS = {"sales": None, "qr_receipts": "txn_id", "purchases": "purchase_id", "expenses": "expense_id",
           "financing_options": "option_id"}


class DataError(ValueError):
    """Raised when records cannot be used. `problems` lists every issue found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class Dataset:
    tables: dict[str, pd.DataFrame]
    settings: dict
    financing_options: list = field(default_factory=list)
    dataset_id: str = "demo"
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> pd.DataFrame:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(name)

    @property
    def as_of(self) -> pd.Timestamp:
        return self.settings["as_of"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _clean_name(name: str) -> str:
    return Path(name).stem.strip().lower().replace(" ", "_").replace("-", "_")


def read_folder(folder: str | Path) -> dict[str, pd.DataFrame]:
    return {_clean_name(p.name): pd.read_csv(p, dtype=str, keep_default_na=False)
            for p in Path(folder).glob("*.csv")}


def read_uploads(files: list[tuple[str, bytes]]) -> dict[str, pd.DataFrame]:
    """files: (filename, content). Accepts .csv files and .xlsx workbooks."""
    raw = {}
    for name, content in files:
        if name.lower().endswith((".xlsx", ".xlsm")):
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, dtype=str, keep_default_na=False)
            raw.update({_clean_name(s): df for s, df in sheets.items()})
        elif name.lower().endswith(".csv"):
            raw[_clean_name(name)] = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        else:
            raise DataError([f"{name}: only .xlsx and .csv files are supported"])
    return raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _rows(mask: pd.Series) -> str:
    rows = [str(i + 2) for i in mask[mask].index[:8]]       # +2: header row, 1-based like Excel
    more = int(mask.sum()) - len(rows)
    return ", ".join(rows) + (f" and {more} more" if more > 0 else "")


def _prepare(name: str, df: pd.DataFrame, problems: list[str], warnings: list[str]) -> pd.DataFrame:
    required, optional, _ = SCHEMA[name]
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"{name}: missing column(s) {', '.join(missing)}")
        return df
    df = df.apply(lambda s: s.astype(str).str.strip() if pd.api.types.is_string_dtype(s) or s.dtype == object else s)
    df = df[~(df[required] == "").all(axis=1)].reset_index(drop=True)          # drop fully blank rows
    for c in required:
        blank = df[c].astype(str) == ""
        if blank.any():
            problems.append(f"{name}: '{c}' is empty in row(s) {_rows(blank)}. Please fill it in.")
    for c, default in optional.items():
        if c not in df.columns:
            df[c] = default
        elif default is not None:
            df[c] = df[c].replace("", default)
    for c in df.columns:
        if c in NUM_COLS:
            raw = df[c]
            df[c] = pd.to_numeric(raw.astype(str).str.replace(",", "").replace("", None), errors="coerce")
            bad = df[c].isna() & (raw.astype(str) != "") & raw.notna()
            if bad.any():
                problems.append(f"{name}: '{c}' is not a number in row(s) {_rows(bad)}")
        elif c in DATE_COLS | TIME_COLS:
            raw = df[c].replace("", None)
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
            bad = parsed.isna() & raw.notna()
            if bad.any():
                problems.append(f"{name}: '{c}' is not a date in row(s) {_rows(bad)}")
            df[c] = parsed.dt.normalize() if c in DATE_COLS else parsed
    return df


def _check_values(t: dict[str, pd.DataFrame], problems: list[str], warnings: list[str]) -> None:
    has = lambda n, c: n in t and c in t[n].columns
    if has("sales", "payment_method"):
        pm = t["sales"].payment_method.str.lower()
        t["sales"]["payment_method"] = pm.replace({"promptpay": "qr", "transfer": "qr", "bank": "qr"})
        bad = ~t["sales"].payment_method.isin(["cash", "qr"])
        if bad.any():
            problems.append(f"sales: payment_method must be cash or qr in row(s) {_rows(bad)}")
    if has("purchases", "has_tax_invoice"):
        inv = t["purchases"].has_tax_invoice.astype(str).str.lower()
        t["purchases"]["has_tax_invoice"] = inv.replace({"y": "yes", "n": "no", "true": "yes", "false": "no"})
        bad = ~t["purchases"].has_tax_invoice.isin(["yes", "no"])
        if bad.any():
            problems.append(f"purchases: has_tax_invoice must be yes or no in row(s) {_rows(bad)}")
    if has("upcoming_bills", "repeat"):
        rep = t["upcoming_bills"]["repeat"].astype(str).str.lower()
        t["upcoming_bills"]["repeat"] = rep
        bad = ~rep.isin(["none", "weekly", "monthly"])
        if bad.any():
            problems.append(f"upcoming_bills: repeat must be none, weekly or monthly in row(s) {_rows(bad)}")
    if has("expenses", "category") and has("expenses", "expense_id"):
        e = t["expenses"]
        e["category"] = e.category.str.lower().str.replace(" ", "_")
        if (e.expense_id.isna() | (e.expense_id.astype(str).isin(['', 'None', 'nan']))).any():
            e["expense_id"] = [x if isinstance(x, str) and x not in ('', 'None', 'nan') else f"E{i + 1:04d}" for i, x in enumerate(e.expense_id)]
    for name, col in ID_COLS.items():
        if col and has(name, col):
            dup = t[name][col].duplicated(keep=False) & (t[name][col].astype(str) != "")
            if dup.any():
                problems.append(f"{name}: '{col}' appears more than once in row(s) {_rows(dup)}. IDs must be unique.")
    for name, col in [("sales", "line_total"), ("qr_receipts", "amount"), ("cash_counts", "counted_cash"),
                      ("purchases", "amount_incl_vat"), ("expenses", "amount"), ("upcoming_bills", "amount")]:
        if has(name, col) and pd.api.types.is_numeric_dtype(t[name][col]):
            neg = t[name][col] < 0
            if neg.any():
                warnings.append(f"{name}: negative '{col}' in row(s) {_rows(neg)}. Check these are refunds or corrections.")
    if has("cash_counts", "date"):
        dup = t["cash_counts"].date.duplicated(keep=False)
        if dup.any():
            problems.append(f"cash_counts: more than one count for the same date in row(s) {_rows(dup)}")


def build_dataset(raw: dict[str, pd.DataFrame], dataset_id: str | None = None) -> Dataset:
    problems, warnings, tables = [], [], {}
    for name in SCHEMA:
        if name not in raw:
            if name not in OPTIONAL_TABLES:
                problems.append(f"Missing record type '{name}' (a sheet or CSV file with that name). {SCHEMA[name][2]}")
            continue
        tables[name] = _prepare(name, raw[name], problems, warnings)
    unknown = sorted(set(raw) - set(SCHEMA) - {"answer_key", "read_me", "readme"})
    if unknown:
        warnings.append(f"Ignored sheets/files not used by BayFlow: {', '.join(unknown)}")
    _check_values(tables, problems, warnings)
    if not problems and len(tables["settings"]) != 1:
        problems.append("settings: must contain exactly one row")
    if problems:
        raise DataError(problems)

    s = tables.pop("settings").iloc[0].to_dict()
    record_dates = [tables["sales"].timestamp.max(), tables["qr_receipts"].timestamp.max(),
                    pd.Timestamp(tables["cash_counts"].date.max()), tables["purchases"].date.max(),
                    tables["expenses"].date.max()]
    as_of = s.get("as_of")
    if as_of is None or pd.isna(as_of):
        as_of = max(pd.Timestamp(d) for d in record_dates if pd.notna(d)).normalize()
    settings = {"opening_date": pd.Timestamp(s["opening_date"]), "opening_balance": float(s["opening_balance"]),
                "minimum_buffer": float(s["minimum_buffer"]), "as_of": pd.Timestamp(as_of),
                "vat_registered": "no" if str(s.get("vat_registered", "yes")).lower() in ("no", "n", "false") else "yes"}
    first = min(pd.Timestamp(d).normalize() for d in [tables["sales"].timestamp.min(), tables["purchases"].date.min()])
    if first < settings["opening_date"]:
        warnings.append(f"Some records are dated before opening_date ({settings['opening_date'].date()}); "
                        "they are not included in the cash balance.")
    fin = tables.pop("financing_options", pd.DataFrame()).to_dict("records")
    if dataset_id is None:
        h = hashlib.sha256()
        for name in sorted(raw):
            h.update(name.encode())
            h.update(raw[name].to_csv(index=False).encode())
        dataset_id = "upload-" + h.hexdigest()[:10]
    return Dataset(tables=tables, settings=settings, financing_options=fin, dataset_id=dataset_id, warnings=warnings)


def load_dataset(folder: str | Path, corrections: list[dict] | None = None, dataset_id: str = "demo") -> Dataset:
    ds = build_dataset(read_folder(folder), dataset_id)
    for c in corrections or []:
        apply_correction_to_dataset(ds, c)
    return ds


def to_excel(folder: str | Path) -> bytes:
    """A workbook in the upload format, used as a template the owner can fill in."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        guide = pd.DataFrame([{"sheet": n, "required columns": ", ".join(r), "optional columns": ", ".join(o),
                               "what it is": d} for n, (r, o, d) in SCHEMA.items()])
        guide.to_excel(xw, sheet_name="read_me", index=False)
        for name, df in read_folder(folder).items():
            if name in SCHEMA:
                df.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Supporting records (so the owner can check a finding themselves)
# ---------------------------------------------------------------------------
def supporting_records(ds: Dataset, evidence: dict, date: str | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    bills = [evidence[k] for k in ("bill_id", "identical_to_bill") if k in evidence]
    if bills:
        out["sales"] = ds.sales[ds.sales.bill_id.isin(bills)]
    if "txn_id" in evidence:
        out["qr_receipts"] = ds.qr_receipts[ds.qr_receipts.txn_id == evidence["txn_id"]]
    if "purchase_id" in evidence:
        out["purchases"] = ds.purchases[ds.purchases.purchase_id == evidence["purchase_id"]]
    if date and ("cash_expected" in evidence or "cash_counted" in evidence):
        d = pd.Timestamp(date)
        out["cash sales that day"] = ds.sales[(ds.sales.timestamp.dt.normalize() == d)
                                              & (ds.sales.payment_method == "cash")]
        out["cash_counts"] = ds.cash_counts[ds.cash_counts.date == d]
    return out


# ---------------------------------------------------------------------------
# Corrections. The agent can propose them; only the owner can approve them.
# Each one edits the loaded copy only, never the owner's file.
# ---------------------------------------------------------------------------
CORRECTION_KINDS = {
    "void_bill": ["bill_id"],
    "set_tax_invoice": ["purchase_id", "has_tax_invoice"],
    "set_cash_count": ["date", "counted_cash"],
}


def describe_current_value(ds: Dataset, kind: str, args: dict):
    """Return the value a correction would replace (shown to the owner)."""
    if kind == "void_bill":
        rows = ds.sales[ds.sales.bill_id == args["bill_id"]]
        if rows.empty:
            raise DataError([f"Bill {args['bill_id']} not found"])
        return {"bill_total": float(rows.line_total.sum()), "lines": len(rows)}
    if kind == "set_tax_invoice":
        if args["has_tax_invoice"] not in ("yes", "no"):
            raise DataError(["has_tax_invoice must be yes or no"])
        m = ds.purchases.purchase_id == args["purchase_id"]
        if not m.any():
            raise DataError([f"Purchase {args['purchase_id']} not found"])
        return {"has_tax_invoice": ds.purchases.loc[m, "has_tax_invoice"].iloc[0]}
    if kind == "set_cash_count":
        m = ds.cash_counts.date == pd.Timestamp(args["date"])
        if not m.any():
            raise DataError([f"No cash count on {args['date']}"])
        float(args["counted_cash"])
        return {"counted_cash": float(ds.cash_counts.loc[m, "counted_cash"].iloc[0])}
    raise DataError([f"Unknown correction kind: {kind}"])


def apply_correction_to_dataset(ds: Dataset, correction: dict) -> None:
    kind, a = correction["kind"], correction["args"]
    describe_current_value(ds, kind, a)  # raises if the target does not exist
    t = ds.tables
    if kind == "void_bill":
        t["sales"] = t["sales"][t["sales"].bill_id != a["bill_id"]].reset_index(drop=True)
    elif kind == "set_tax_invoice":
        t["purchases"].loc[t["purchases"].purchase_id == a["purchase_id"], "has_tax_invoice"] = a["has_tax_invoice"]
    elif kind == "set_cash_count":
        t["cash_counts"].loc[t["cash_counts"].date == pd.Timestamp(a["date"]), "counted_cash"] = float(a["counted_cash"])
