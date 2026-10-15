"""Load and validate the owner's records.

Every record type is a plain CSV (or one sheet of an Excel file) with the
columns listed in SCHEMA. Original files are never modified: approved
corrections are kept separately and applied on top when the data is loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA: dict[str, list[str]] = {
    "sales": ["bill_id", "timestamp", "item_id", "qty", "unit_price", "line_total", "payment_method"],
    "qr_receipts": ["txn_id", "timestamp", "amount", "reference"],
    "cash_counts": ["date", "counted_cash"],
    "purchases": ["purchase_id", "date", "supplier", "ingredient_id", "qty", "amount_incl_vat",
                  "vat_amount", "has_tax_invoice", "note"],
    "stock_counts": ["date", "ingredient_id", "closing_count"],
    "expenses": ["expense_id", "date", "category", "description", "amount"],
    "upcoming_bills": ["due_date", "category", "description", "amount"],
    "menu": ["item_id", "item_name", "price_thb"],
    "recipes": ["item_id", "ingredient_id", "qty_per_item"],
    "ingredients": ["ingredient_id", "name", "unit", "unit_cost_thb", "waste_allowance_pct"],
}
JSON_FILES = ("cash_position", "financing_options")


class DataError(ValueError):
    """Raised when a file is missing or has the wrong columns."""


@dataclass
class Dataset:
    tables: dict[str, pd.DataFrame]
    cash_position: dict
    financing_options: list = field(default_factory=list)

    def __getattr__(self, name: str) -> pd.DataFrame:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(name)


def _read_table(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"note": str, "bill_id": str, "txn_id": str})
    missing = [c for c in SCHEMA[name] if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {', '.join(missing)}")
    df = df[SCHEMA[name]].copy()
    if "note" in df:
        df["note"] = df["note"].fillna("")
    for col in ("timestamp",):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    for col in ("date", "due_date"):
        if col in df:
            df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df


def load_dataset(folder: str | Path, corrections: list[dict] | None = None) -> Dataset:
    folder = Path(folder)
    tables = {}
    for name in SCHEMA:
        path = folder / f"{name}.csv"
        if not path.exists():
            raise DataError(f"Missing file: {name}.csv")
        tables[name] = _read_table(path, name)
    position = json.loads((folder / "cash_position.json").read_text())
    fin_path = folder / "financing_options.json"
    financing = json.loads(fin_path.read_text()) if fin_path.exists() else []
    ds = Dataset(tables=tables, cash_position=position, financing_options=financing)
    for c in corrections or []:
        apply_correction_to_dataset(ds, c)
    return ds


# ---- corrections -----------------------------------------------------------
# Only these kinds of change exist. The agent can propose them; only the owner
# can approve them (see permissions.py). Each one edits the loaded copy only.
CORRECTION_KINDS = {
    "void_bill": ["bill_id"],
    "set_stock_count": ["date", "ingredient_id", "closing_count"],
    "set_tax_invoice": ["purchase_id", "has_tax_invoice"],
    "set_cash_count": ["date", "counted_cash"],
}


def describe_current_value(ds: Dataset, kind: str, args: dict):
    """Return the value a correction would replace (shown to the owner)."""
    if kind == "void_bill":
        rows = ds.sales[ds.sales.bill_id == args["bill_id"]]
        if rows.empty:
            raise DataError(f"Bill {args['bill_id']} not found")
        return {"bill_total": float(rows.line_total.sum()), "lines": len(rows)}
    if kind == "set_stock_count":
        m = (ds.stock_counts.date == pd.Timestamp(args["date"])) & (ds.stock_counts.ingredient_id == args["ingredient_id"])
        if not m.any():
            raise DataError("Stock count row not found")
        return {"closing_count": float(ds.stock_counts.loc[m, "closing_count"].iloc[0])}
    if kind == "set_tax_invoice":
        m = ds.purchases.purchase_id == args["purchase_id"]
        if not m.any():
            raise DataError(f"Purchase {args['purchase_id']} not found")
        return {"has_tax_invoice": ds.purchases.loc[m, "has_tax_invoice"].iloc[0]}
    if kind == "set_cash_count":
        m = ds.cash_counts.date == pd.Timestamp(args["date"])
        if not m.any():
            raise DataError("Cash count row not found")
        return {"counted_cash": float(ds.cash_counts.loc[m, "counted_cash"].iloc[0])}
    raise DataError(f"Unknown correction kind: {kind}")


def apply_correction_to_dataset(ds: Dataset, correction: dict) -> None:
    kind, a = correction["kind"], correction["args"]
    describe_current_value(ds, kind, a)  # raises if the target does not exist
    t = ds.tables
    if kind == "void_bill":
        t["sales"] = t["sales"][t["sales"].bill_id != a["bill_id"]].reset_index(drop=True)
    elif kind == "set_stock_count":
        m = (t["stock_counts"].date == pd.Timestamp(a["date"])) & (t["stock_counts"].ingredient_id == a["ingredient_id"])
        t["stock_counts"].loc[m, "closing_count"] = float(a["closing_count"])
    elif kind == "set_tax_invoice":
        t["purchases"].loc[t["purchases"].purchase_id == a["purchase_id"], "has_tax_invoice"] = a["has_tax_invoice"]
    elif kind == "set_cash_count":
        t["cash_counts"].loc[t["cash_counts"].date == pd.Timestamp(a["date"]), "counted_cash"] = float(a["counted_cash"])
