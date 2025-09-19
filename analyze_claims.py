#!/usr/bin/env python3
"""
analyze_claims.py

A small, portfolio-ready analyzer for Kaggle-style health insurance claims CSVs.

What it does:
- Loads a CSV (or Excel) of claims data
- Cleans obvious issues (blanks-only rows/cols, leading/trailing spaces, duplicated rows)
- Normalizes common date columns to ISO (YYYY-MM-DD)
- Produces summary stats (overall and grouped) and saves them to CSV
- Generates simple charts (saved to ./outputs)

Designed to be flexible with column names commonly seen in Kaggle datasets:
- claim id:        claim_id, id, claim_number
- patient id:      patient_id, customer_id, client_id
- provider:        provider, provider_name, hospital
- payer:           payer, insurer, insurance, company
- amount:          amount, billed_amount, claim_amount, total_claim_amount, charges
- status:          status, claim_status, outcome, decision
- dates:           dos, date_of_service, service_date, claim_date, incident_date, report_date, submission_date, received_date

Run:
    python analyze_claims.py insurance_claims.csv
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

# ---- Configuration ----
DEFAULT_OUT_DIR = Path("outputs")
DEFAULT_SUMMARY_PREFIX = "summary_"
DEFAULT_CLEAN_PREFIX = "cleaned_"

# Common column name candidates for flexible mapping
CANDIDATES = {
    "claim_id": ["claim_id", "id", "claim_number", "claimno", "claim no", "claim"],
    "patient_id": ["patient_id", "customer_id", "client_id", "member_id", "patient"],
    "provider": ["provider", "provider_name", "hospital", "facility", "clinic"],
    "payer": ["payer", "insurer", "insurance", "company", "payor"],
    "amount": [
        "amount",
        "billed_amount",
        "claim_amount",
        "total_claim_amount",
        "charges",
        "charge_amount",
        "paid_amount",
        "approved_amount",
    ],
    "status": ["status", "claim_status", "outcome", "decision", "state"],
}

DATE_PATTERNS = re.compile(
    r"\b(dos|date_of_service|service_date|claim_date|incident_date|report_date|submission_date|received_date|date)\b",
    re.IGNORECASE,
)


# ---- Utilities ----
def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _infer_col(df: pd.DataFrame, options: Iterable[str]) -> Optional[str]:
    """Return the first matching column (case-insensitive), or None."""
    lowered = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lowered:
            return lowered[opt.lower()]
    return None


def _find_date_columns(df: pd.DataFrame) -> List[str]:
    """Heuristically find date-like columns via name patterns."""
    date_cols = []
    for c in df.columns:
        if DATE_PATTERNS.search(c):
            date_cols.append(c)
    # de-dup while preserving order
    seen = set()
    out = []
    for c in date_cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _coerce_dates(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce listed columns to ISO date strings where possible."""
    for c in cols:
        try:
            ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True, utc=False)
            # Keep only date part (no time zone/time-of-day)
            df[c] = ser.dt.date.astype("string")
        except Exception:
            # Leave column as is on failure
            pass
    return df


def _safe_float(series: pd.Series) -> pd.Series:
    """Try to coerce to float; if it fails, return the original."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series


# ---- Core Steps ----
def load_table(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    # Drop completely empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Trim whitespace from string-like columns
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"• Removed {before - after} duplicate row(s)")

    return df


def normalize_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    """
    Map flexible column names into a normalized schema dict for downstream logic.
    Returns: (df, mapping, date_columns)
      mapping: canonical_name -> actual_column_name (or None if not found)
      date_columns: list of detected date-like columns
    """
    mapping = {
        "claim_id": _infer_col(df, CANDIDATES["claim_id"]),
        "patient_id": _infer_col(df, CANDIDATES["patient_id"]),
        "provider": _infer_col(df, CANDIDATES["provider"]),
        "payer": _infer_col(df, CANDIDATES["payer"]),
        "amount": _infer_col(df, CANDIDATES["amount"]),
        "status": _infer_col(df, CANDIDATES["status"]),
    }

    date_cols = _find_date_columns(df)
    df = _coerce_dates(df, date_cols)

    # Coerce amount if present
    if mapping["amount"]:
        df[mapping["amount"]] = _safe_float(df[mapping["amount"]])

    return df, mapping, date_cols


def summarize(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    group_by: Iterable[str],
) -> pd.DataFrame:
    """
    Build a tidy summary table consisting of:
      - overall totals
      - grouped totals/averages for requested fields (provider/payer/status/etc.)
    """
    rows = []

    # overall
    amount_col = mapping.get("amount")
    overall = {
        "group_type": "overall",
        "group_value": "ALL",
        "claims_count": len(df),
        "total_amount": float(df[amount_col].sum()) if amount_col else None,
        "avg_amount": float(df[amount_col].mean()) if amount_col else None,
    }
    rows.append(overall)

    # grouped
    canonical_to_actual = {k: v for k, v in mapping.items() if v}
    # allow grouping by canonical keys or actual column names
    actual_group_cols = []
    for g in group_by:
        # try canonical → actual
        actual = canonical_to_actual.get(g)
        if actual is None and g in df.columns:
            actual = g
        if actual is not None and actual not in actual_group_cols:
            actual_group_cols.append(actual)

    for col in actual_group_cols:
        grp = df.groupby(col, dropna=False)
        tmp = grp[amount_col].agg(["count", "sum", "mean"]) if amount_col else grp.size().to_frame("count")
        tmp = tmp.reset_index()
        for _, r in tmp.iterrows():
            rows.append(
                {
                    "group_type": col,
                    "group_value": r[col] if pd.notna(r[col]) else "(missing)",
                    "claims_count": int(r["count"]) if "count" in r else int(r[0]),
                    "total_amount": float(r["sum"]) if "sum" in r and pd.notna(r["sum"]) else None,
                    "avg_amount": float(r["mean"]) if "mean" in r and pd.notna(r["mean"]) else None,
                }
            )

    return pd.DataFrame(rows)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_clean_and_summary(
    *,
    df_clean: pd.DataFrame,
    summary: pd.DataFrame,
    inp_path: Path,
    out_clean: Optional[Path],
    out_summary: Optional[Path],
) -> Tuple[Path, Path]:
    """Resolve filenames and write CSVs. Return written paths."""
    if out_clean is None:
        out_clean = inp_path.with_name(f"{DEFAULT_CLEAN_PREFIX}{inp_path.stem}.csv")
    if out_summary is None:
        out_summary = inp_path.with_name(f"{DEFAULT_SUMMARY_PREFIX}{inp_path.stem}.csv")

    df_clean.to_csv(out_clean, index=False)
    summary.to_csv(out_summary, index=False)
    return out_clean, out_summary


def plot_status_distribution(df: pd.DataFrame, status_col: Optional[str], out_dir: Path) -> Optional[Path]:
    if not status_col or status_col not in df.columns:
        return None

    counts = df[status_col].fillna("(missing)").value_counts().sort_values(ascending=False)
    if counts.empty:
        return None

    ensure_out_dir(out_dir)
    out_path = out_dir / "claim_status_distribution.png"

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Claim Status Distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


# ---- CLI ----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze (clean + summarize + chart) Kaggle-style insurance claims."
    )
    p.add_argument(
        "input",
        help="Path to CSV/XLSX (e.g., insurance_claims.csv)",
    )
    p.add_argument(
        "--sheet",
        default="",
        help="Excel sheet name (if reading .xlsx). Default: first sheet.",
    )
    p.add_argument(
        "--by",
        nargs="*",
        default=["provider", "payer", "status"],
        help="Group-by fields (canonical or actual column names). Default: provider payer status.",
    )
    p.add_argument(
        "--out-clean",
        default="",
        help="Output CSV for cleaned data. Default: cleaned_<input>.csv",
    )
    p.add_argument(
        "--out-summary",
        default="",
        help="Output CSV for summary table. Default: summary_<input>.csv",
    )
    p.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable chart generation.",
    )
    p.add_argument(
        "--outputs-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory to save charts. Default: {DEFAULT_OUT_DIR}",
    )
    return p


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    inp = Path(args.input)
    sheet = args.sheet or None
    out_clean = Path(args.out_clean) if args.out_clean else None
    out_summary = Path(args.out_summary) if args.out_summary else None
    out_dir = Path(args.outputs_dir)

    _print_header("Loading data")
    try:
        df = load_table(inp, sheet)
    except Exception as e:
        print(f"✗ Failed to load input: {e}")
        return 2

    print(f"• Rows: {len(df):,}  Cols: {len(df.columns)}")
    print(f"• Columns: {list(df.columns)}")

    _print_header("Normalizing schema")
    df, mapping, date_cols = normalize_schema(df)
    print("• Column mapping (canonical → actual):")
    for k, v in mapping.items():
        print(f"   - {k:10s} → {v!r}")
    if date_cols:
        print(f"• Date columns normalized: {', '.join(date_cols)}")
    else:
        print("• Date columns normalized: (none detected)")

    _print_header("Summarizing")
    summary = summarize(df, mapping, args.by)
    print(f"• Summary rows: {len(summary):,}")

    _print_header("Saving CSVs")
    try:
        clean_path, summary_path = save_clean_and_summary(
            df_clean=df, summary=summary, inp_path=inp, out_clean=out_clean, out_summary=out_summary
        )
        print(f"• Cleaned CSV : {clean_path}")
        print(f"• Summary CSV : {summary_path}")
    except Exception as e:
        print(f"✗ Failed to write outputs: {e}")
        return 3

    chart_path = None
    if not args.no_charts:
        _print_header("Plotting")
        try:
            chart_path = plot_status_distribution(df, mapping.get("status"), out_dir)
            if chart_path:
                print(f"• Chart saved : {chart_path}")
            else:
                print("• Chart skipped: no usable status column found.")
        except Exception as e:
            print(f"! Chart step failed (continuing): {e}")

    _print_header("Done")
    # A friendly, human-readable epilogue similar to your README example
    print("✅ Claims Analysis Complete")
    print(f"• Input         : {inp.name}")
    print(f"• Cleaned CSV   : {clean_path.name}")
    print(f"• Summary CSV   : {summary_path.name}")
    print(f"• Rows          : {len(df):,}")
    amt_col = mapping.get("amount")
    if amt_col and amt_col in df.columns:
        total_amt = pd.to_numeric(df[amt_col], errors="coerce").sum()
        avg_amt = pd.to_numeric(df[amt_col], errors="coerce").mean()
        print(f"• Total Amount  : {total_amt:,.2f}")
        print(f"• Avg Amount    : {avg_amt:,.2f}")
    if chart_path:
        print(f"• Chart         : {chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())