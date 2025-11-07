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


def read_out_file(path: str) -> pd.DataFrame:
    print(f"Reading .out file: {path}")
    with open(path, 'r') as f:
        lines = f.readlines()

    headers = []
    data = []
    in_data = False
    in_fields = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == 'START-OF-FIELDS':
            in_fields = True
            continue
        elif line == 'END-OF-FIELDS':
            in_fields = False
            continue
        elif line == 'START-OF-DATA':
            in_data = True
            continue
        elif line == 'END-OF-DATA':
            in_data = False
            continue

        if in_fields:
            headers.append(line)
        elif in_data:
            row_data = [val.strip() for val in line.split('|')]
            data.append(row_data)

    num_headers = len(headers)
    # Determine metadata offset: columns at left that are filled in every row
    if not data:
        raise ValueError(f"{path}: No data rows found")

    max_columns = max(len(row) for row in data)
    column_stats = [0] * max_columns
    for row in data:
        for i, val in enumerate(row):
            if val.strip():
                column_stats[i] += 1

    offset = 0
    for count in column_stats:
        if count == len(data):
            offset += 1
        else:
            break

    # Build columns: metadata columns (KEY, META2, ...) + headers
    meta_cols = [f"META{i+1}" for i in range(offset)]
    if meta_cols:
        meta_cols[0] = "KEY"
    columns = meta_cols + headers

    # Normalize rows to expected length
    expected_length = offset + len(headers)
    processed_data = []
    for row in data:
        if len(row) < expected_length:
            row = row + [''] * (expected_length - len(row))
        elif len(row) > expected_length:
            row = row[:expected_length]
        processed_data.append(row)

    df = pd.DataFrame(processed_data, columns=columns)
    df = df.fillna("")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from: {path}")
    return df

def get_metadata_columns(df: pd.DataFrame) -> List[str]:
    # Only treat the key column (first column) as metadata
    return [df.columns[0]] if len(df.columns) > 0 else []


def col_letter_to_index(letter: str) -> int:
    """Convert Excel-style column letters (A, B, ..., Z, AA, AB, ...) to 0-based index."""
    letter = letter.strip().upper()
    if not letter:
        raise ValueError("Empty column letter")
    idx = 0
    for ch in letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"Invalid column letter: {letter}")
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def parse_ignore_columns(df: pd.DataFrame, ignore_str: str) -> Set[str]:
    """Parse a comma-separated list of column letters and return a set of column names to ignore.
    If a specified index is out of range it will be skipped with a warning.
    """
    ignore_cols: Set[str] = set()
    if not ignore_str:
        return ignore_cols
    for token in [t.strip() for t in ignore_str.split(',') if t.strip()]:
        try:
            idx = col_letter_to_index(token)
        except ValueError as e:
            print(f"Warning: ignoring invalid column token '{token}': {e}")
            continue
        if idx < 0 or idx >= len(df.columns):
            print(f"Warning: column '{token}' index {idx} out of range for dataframe with {len(df.columns)} columns; skipping")
            continue
        ignore_cols.add(df.columns[idx])
    return ignore_cols


def get_key_column_name(df: pd.DataFrame) -> str:
    # Always treat the first column (column A) as the key column.
    if len(df.columns) == 0:
        raise ValueError("Input file has no columns")
    return df.columns[0]


def intersect_columns(df1: pd.DataFrame, df2: pd.DataFrame, exclude: Set[str]) -> List[str]:
    return [c for c in df1.columns if c in df2.columns and c not in exclude]


def compare_rows_fast(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col1: str,
    id_col2: str,
    metadata_cols1: List[str],
    metadata_cols2: List[str],
    ignore_cols1: Set[str] = None,
    ignore_cols2: Set[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Set[str], Set[str], List[str]]:
    print("Comparing rows by KEY (stream-fast)...")
    start_ts = time.perf_counter()

    def _is_sorted(series: pd.Series) -> bool:
        # Attempt to convert to numeric. If errors, coerce to NaN.
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Check if all values could be converted to numeric
        if not numeric_series.isnull().any():
            # If all numeric, compare numerically
            vals = numeric_series.to_numpy()
            return np.all(vals[1:] >= vals[:-1]) if len(vals) > 1 else True
        else:
            # If not all numeric, compare as strings
            vals = series.astype(str).to_numpy()
            return np.all(vals[1:] >= vals[:-1]) if len(vals) > 1 else True

    ignore_cols1 = ignore_cols1 or set()
    ignore_cols2 = ignore_cols2 or set()

    exclude_cols = {id_col1, id_col2}.union(metadata_cols1).union(metadata_cols2).union(ignore_cols1).union(ignore_cols2)
    comparable_cols = intersect_columns(df1, df2, exclude=exclude_cols)

    ids1 = df1[id_col1].to_numpy()
    ids2 = df2[id_col2].to_numpy()

    if not (_is_sorted(df1[id_col1]) and _is_sorted(df2[id_col2])):
        print("Warning: KEYs not detected as ascending in one or both files; sorting for fast path...")
        # Sort both frames by KEY numerically to enable fast path
        df1 = df1.assign(_key_num=pd.to_numeric(df1[id_col1].astype(str).str.strip(), errors="coerce")).sort_values(by="_key_num", kind="stable").drop(columns=["_key_num"]).reset_index(drop=True)
        df2 = df2.assign(_key_num=pd.to_numeric(df2[id_col2].astype(str).str.strip(), errors="coerce")).sort_values(by="_key_num", kind="stable").drop(columns=["_key_num"]).reset_index(drop=True)
        ids1 = df1[id_col1].to_numpy()
        ids2 = df2[id_col2].to_numpy()

    # Use numeric arrays for ordering comparisons
    ids1_num = pd.to_numeric(pd.Series(ids1, dtype=str).str.strip(), errors="coerce").to_numpy()
    ids2_num = pd.to_numeric(pd.Series(ids2, dtype=str).str.strip(), errors="coerce").to_numpy()

    def normalize_key(val) -> str:
        """Normalize KEY to consistent string format."""
        s = str(val).strip()
        try:
            num_val = float(s)
            if num_val.is_integer():
                return str(int(num_val))
        except (ValueError, TypeError):
            pass
        return s

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
            s1 = df1.iloc[i][comparable_cols]
            s2 = df2.iloc[j][comparable_cols]
            diff_cols: List[str] = []
            for col in comparable_cols:
                if str(s1[col]) != str(s2[col]):
                    diff_cols.append(col)
            if diff_cols:
                id1_str = normalize_key(id1)
                id2_str = normalize_key(id2)
                diffs1[id1_str] = diff_cols
                diffs2[id2_str] = diff_cols
            i += 1
            j += 1
        elif n1 < n2:
            only_in_1.add(normalize_key(id1))
            i += 1
        else:
            only_in_2.add(normalize_key(id2))
            j += 1

    while i < len1:
        only_in_1.add(normalize_key(ids1[i]))
        i += 1
    while j < len2:
        only_in_2.add(normalize_key(ids2[j]))
        j += 1

    common_ids_count = (len(df1) - len(only_in_1)) if len(df1) <= len(df2) else (len(df2) - len(only_in_2))
    print(
        "Comparison summary: "
        f"common KEYs={common_ids_count}, differing rows={len(diffs1)}, "
        f"only in file1={len(only_in_1)}, only in file2={len(only_in_2)}"
    )
    print_timing("Compare elapsed", start_ts)

    return diffs1, diffs2, only_in_1, only_in_2, comparable_cols


def write_stream_highlight(
    df: pd.DataFrame,
    out_path: str,
    key_to_diff_cols: Dict[str, List[str]],
    keys_only_in_this: Set[str],
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

    for c, name in enumerate(df.columns):
        ws.write(0, c, name, fmt_header)

    def normalize_key(val) -> str:
        """Normalize KEY to consistent string format (same as in comparison)."""
        s = str(val).strip()
        try:
            num_val = float(s)
            if num_val.is_integer():
                return str(int(num_val))
        except (ValueError, TypeError):
            pass
        return s

    # Write data rows
    for r in range(len(df)):
        row = df.iloc[r]
        # Normalize KEY string to match how it was stored during comparison
        key = normalize_key(row.iloc[0])
        
        # Determine row highlighting: blue (only in this file) takes priority over yellow (has differences)
        is_only_in_this = key in keys_only_in_this
        diff_cols = key_to_diff_cols.get(key, [])

        # Write each cell with appropriate format
        for c, value in enumerate(row.values):
            col_name = str(df.columns[c])
            
            if is_only_in_this:
                # Blue row for KEYs only in this file
                ws.write(r + 1, c, value, fmt_row_blue)
            elif diff_cols:
                # Row has differences: yellow background for all cells
                if col_name in diff_cols:
                    # Red cell for differing cells (red overrides yellow)
                    ws.write(r + 1, c, value, fmt_cell_red)
                else:
                    # Yellow cell for non-differing cells in a row with differences
                    ws.write(r + 1, c, value, fmt_row_yellow)
            else:
                # Normal cell, no highlighting
                ws.write(r + 1, c, value)

    wb.close()
    print_timing("Write elapsed", start_ts)


def derive_output_paths(file1: str, file2: str, output_dir: str) -> Tuple[str, str]:
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    out1 = os.path.join(output_dir, f"{base1}__comp.xlsx")
    out2 = os.path.join(output_dir, f"{base2}__comp.xlsx")
    return out1, out2


def _auto_discover_input_files(input_dir: str) -> Tuple[str, str]:
    candidates: List[str] = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            if name.lower().endswith(".out") and not name.startswith("~"):
                candidates.append(os.path.join(input_dir, name))
    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Expected at least two .out files in '{input_dir}'. Found: {len(candidates)}"
        )
    return candidates[0], candidates[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "FAST: Compare two .out files by KEY (assumed ascending or will be sorted). "
            "Stream-write new highlighted Excel files: differing rows yellow, differing cells red, unmatched rows blue."
        )
    )
    parser.add_argument(
        "file1",
        nargs="?",
        help="Path to first .out file. If omitted, auto-uses two files from ./input",
    )
    parser.add_argument(
        "file2",
        nargs="?",
        help="Path to second .out file. If omitted, auto-uses two files from ./input",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join("input"),
        help="Directory to read input .out files when auto-discovering (default: input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join("output"),
        help="Directory to write highlighted Excel files (default: output)",
    )
    parser.add_argument(
        "--ignore-columns",
        "-x",
        default="",
        help="Comma-separated Excel column letters to ignore in comparison (e.g. A,AB,AC)",
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
        df1 = read_out_file(file1)
        df2 = read_out_file(file2)
        print_timing("Read total", t0)
    except Exception as e:
        print(f"Failed to read .out files: {e}")
        return 1

    id_col1 = get_key_column_name(df1)
    id_col2 = get_key_column_name(df2)
    print(f"Detected KEY columns -> file1: '{id_col1}', file2: '{id_col2}'")

    metadata_cols1 = get_metadata_columns(df1)
    metadata_cols2 = get_metadata_columns(df2)
    print(f"Metadata columns in file1: {metadata_cols1}")
    print(f"Metadata columns in file2: {metadata_cols2}")

    # Parse ignore-columns option and map to actual column names
    ignore_cols1 = parse_ignore_columns(df1, args.ignore_columns)
    ignore_cols2 = parse_ignore_columns(df2, args.ignore_columns)
    if ignore_cols1 or ignore_cols2:
        print(f"Ignoring columns for comparison (file1): {sorted(ignore_cols1)}")
        print(f"Ignoring columns for comparison (file2): {sorted(ignore_cols2)}")

    diffs1, diffs2, only_in_1, only_in_2, comparable_cols = compare_rows_fast(
        df1,
        df2,
        id_col1,
        id_col2,
        metadata_cols1,
        metadata_cols2,
        ignore_cols1=ignore_cols1,
        ignore_cols2=ignore_cols2,
    )

    out1, out2 = derive_output_paths(file1, file2, args.output)

    try:
        write_stream_highlight(df1, out1, diffs1, only_in_1)
        write_stream_highlight(df2, out2, diffs2, only_in_2)
    except Exception as e:
        print(f"Failed to write highlighted workbooks: {e}")
        return 1

    print(f"Done. Wrote: {out1}")
    print(f"Done. Wrote: {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())