import argparse
import os
import sys
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import time


def print_timing(label: str, start_ts: float) -> None:
    elapsed = time.perf_counter() - start_ts
    print(f"{label}: {elapsed:.2f}s")


def read_excel_as_str(path: str) -> pd.DataFrame:
    print(f"Reading Excel: {path}")
    df = pd.read_excel(path, dtype=str)
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from: {path}")
    return df


def get_sec_id_column_name(df: pd.DataFrame) -> str:
    if len(df.columns) == 0:
        raise ValueError("Excel has no columns")
    for col in df.columns:
        if str(col).strip().lower() == "sec_id":
            return col
    return df.columns[0]


def intersect_columns(df1: pd.DataFrame, df2: pd.DataFrame, exclude: Set[str]) -> List[str]:
    return [c for c in df1.columns if c in df2.columns and c not in exclude]


def compare_rows_fast(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col1: str,
    id_col2: str,
    assume_sorted: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Set[str], Set[str], List[str]]:
    print("Comparing rows by SEC_ID (stream-fast)...")
    start_ts = time.perf_counter()

    def _is_sorted(series: pd.Series) -> bool:
        # Robust monotonic check: prefer numeric if all digits, else string order
        if len(series) <= 1:
            return True
        s = series.astype(str).str.strip()
        all_digits = s.str.fullmatch(r"\d+").all()
        if all_digits:
            nums = pd.to_numeric(s, errors="raise")
            return nums.is_monotonic_increasing
        # Fallback to string-based
        return s.is_monotonic_increasing

    comparable_cols = intersect_columns(df1, df2, exclude={id_col1, id_col2})

    ids1 = df1[id_col1].to_numpy()
    ids2 = df2[id_col2].to_numpy()

    if not assume_sorted and not (_is_sorted(df1[id_col1]) and _is_sorted(df2[id_col2])):
        print("Warning: SEC_IDs not detected as ascending in one or both files; sorting for fast path...")
        # Sort both frames by SEC_ID numerically to enable fast path
        df1 = df1.assign(_sec_id_num=pd.to_numeric(df1[id_col1].astype(str).str.strip(), errors="coerce")).sort_values(by="_sec_id_num", kind="stable").drop(columns=["_sec_id_num"]).reset_index(drop=True)
        df2 = df2.assign(_sec_id_num=pd.to_numeric(df2[id_col2].astype(str).str.strip(), errors="coerce")).sort_values(by="_sec_id_num", kind="stable").drop(columns=["_sec_id_num"]).reset_index(drop=True)
        ids1 = df1[id_col1].to_numpy()
        ids2 = df2[id_col2].to_numpy()

    # Use numeric arrays for ordering comparisons
    ids1_num = pd.to_numeric(pd.Series(ids1, dtype=str).str.strip(), errors="coerce").to_numpy()
    ids2_num = pd.to_numeric(pd.Series(ids2, dtype=str).str.strip(), errors="coerce").to_numpy()

    only_in_1: Set[str] = set()
    only_in_2: Set[str] = set()
    diffs1: Dict[str, List[str]] = {}
    diffs2: Dict[str, List[str]] = {}

    i = 0
    j = 0
    len1 = len(ids1)
    len2 = len(ids2)
    while i < len1 and j < len2:
        id1 = ids1[i]
        id2 = ids2[j]
        n1 = ids1_num[i]
        n2 = ids2_num[j]
        if n1 == n2:
            # Per-row comparison to avoid building huge 2D string arrays in memory
            s1 = df1.iloc[i][comparable_cols]
            s2 = df2.iloc[j][comparable_cols]
            diff_cols: List[str] = []
            for col in comparable_cols:
                if str(s1[col]) != str(s2[col]):
                    diff_cols.append(col)
            if diff_cols:
                diffs1[str(id1)] = diff_cols
                diffs2[str(id2)] = diff_cols
            i += 1
            j += 1
        elif n1 < n2:
            only_in_1.add(str(id1))
            i += 1
        else:
            only_in_2.add(str(id2))
            j += 1

    while i < len1:
        only_in_1.add(str(ids1[i]))
        i += 1
    while j < len2:
        only_in_2.add(str(ids2[j]))
        j += 1

    common_ids_count = (len(df1) - len(only_in_1)) if len(df1) <= len(df2) else (len(df2) - len(only_in_2))
    print(
        "Comparison summary: "
        f"common SEC_IDs={common_ids_count}, differing rows={len(diffs1)}, "
        f"only in file1={len(only_in_1)}, only in file2={len(only_in_2)}"
    )
    print_timing("Compare elapsed", start_ts)

    return diffs1, diffs2, only_in_1, only_in_2, comparable_cols


def write_stream_highlight(
    df: pd.DataFrame,
    out_path: str,
    sec_id_to_diff_cols: Dict[str, List[str]],
    sec_ids_only_in_this: Set[str],
    id_col_name: str,
):
    import xlsxwriter

    start_ts = time.perf_counter()
    print(f"Stream writing highlighted workbook to: {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb = xlsxwriter.Workbook(out_path)
    ws = wb.add_worksheet("Sheet1")

    fmt_header = wb.add_format({"bold": True})
    fmt_row_yellow = wb.add_format({"bg_color": "#FFFF99"})
    fmt_cell_red = wb.add_format({"bg_color": "#FF9999"})
    fmt_row_blue = wb.add_format({"bg_color": "#99CCFF"})

    # Header
    for c, name in enumerate(df.columns):
        ws.write(0, c, name, fmt_header)

    # Map column name to index
    col_to_idx = {str(col): idx for idx, col in enumerate(df.columns)}

    # Write data rows
    for r in range(len(df)):
        row = df.iloc[r]
        sec_id = str(row[id_col_name])
        row_format = None
        diff_cols: List[str] = []

        if sec_id in sec_ids_only_in_this:
            row_format = fmt_row_blue
        elif sec_id in sec_id_to_diff_cols:
            row_format = fmt_row_yellow
            diff_cols = sec_id_to_diff_cols[sec_id]

        if row_format is not None:
            ws.set_row(r + 1, None, row_format)

        # Write all cells with default format (row format applies if set)
        for c, value in enumerate(row.values):
            ws.write(r + 1, c, value)

        # Overwrite only diff cells with red format
        if diff_cols:
            for col_name in diff_cols:
                cidx = col_to_idx.get(col_name)
                if cidx is not None:
                    ws.write(r + 1, cidx, row.iloc[cidx], fmt_cell_red)

    wb.close()
    print_timing("Write elapsed", start_ts)


def derive_output_paths(file1: str, file2: str, output_dir: str) -> Tuple[str, str]:
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    out1 = os.path.join(output_dir, f"{base1}__stream.xlsx")
    out2 = os.path.join(output_dir, f"{base2}__stream.xlsx")
    return out1, out2


def _auto_discover_input_files(input_dir: str) -> Tuple[str, str]:
    candidates: List[str] = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            if name.lower().endswith(".xlsx") and not name.startswith("~$"):
                candidates.append(os.path.join(input_dir, name))
    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Expected at least two .xlsx files in '{input_dir}'. Found: {len(candidates)}"
        )
    return candidates[0], candidates[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FAST: Compare two Excel files by SEC_ID (assumed ascending or will be sorted). "
            "Stream-write new highlighted files: differing rows yellow, differing cells red, unmatched rows blue."
        )
    )
    parser.add_argument(
        "file1",
        nargs="?",
        help="Path to first Excel file (.xlsx). If omitted, auto-uses two files from ./input",
    )
    parser.add_argument(
        "file2",
        nargs="?",
        help="Path to second Excel file (.xlsx). If omitted, auto-uses two files from ./input",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join("input"),
        help="Directory to read input Excel files when auto-discovering (default: input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join("output"),
        help="Directory to write highlighted Excel files (default: output)",
    )
    parser.add_argument(
        "--assume-sorted",
        action="store_true",
        help="Assume SEC_ID columns are already ascending; skip checks and any sorting",
    )

    args = parser.parse_args()

    if args.file1 and args.file2:
        file1 = args.file1
        file2 = args.file2
    else:
        try:
            file1, file2 = _auto_discover_input_files(args.input)
            print(f"Auto-discovered input files: {file1} | {file2}")
        except Exception as e:
            print(f"Input discovery error: {e}")
            return 1

    try:
        t0 = time.perf_counter()
        df1 = read_excel_as_str(file1)
        df2 = read_excel_as_str(file2)
        print_timing("Read total", t0)
    except Exception as e:
        print(f"Failed to read Excel files: {e}")
        return 1

    id_col1 = get_sec_id_column_name(df1)
    id_col2 = get_sec_id_column_name(df2)
    print(f"Detected SEC_ID columns -> file1: '{id_col1}', file2: '{id_col2}'")

    diffs1, diffs2, only_in_1, only_in_2, comparable_cols = compare_rows_fast(
        df1, df2, id_col1, id_col2, assume_sorted=args.assume_sorted
    )

    out1, out2 = derive_output_paths(file1, file2, args.output)

    try:
        write_stream_highlight(df1, out1, diffs1, only_in_1, id_col1)
        write_stream_highlight(df2, out2, diffs2, only_in_2, id_col2)
    except Exception as e:
        print(f"Failed to write highlighted workbooks: {e}")
        return 1

    print(f"Done. Wrote: {out1}")
    print(f"Done. Wrote: {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


