import argparse
import os
import sys
from typing import Dict, List, Set, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


RED_FILL = PatternFill(start_color="FFFF9999", end_color="FFFF9999", fill_type="solid")
YELLOW_FILL = PatternFill(start_color="FFFFFF99", end_color="FFFFFF99", fill_type="solid")
BLUE_FILL = PatternFill(start_color="FF99CCFF", end_color="FF99CCFF", fill_type="solid")


def read_excel_as_str(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    # Normalize whitespace and NaNs to empty strings for stable comparison
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def get_sec_id_column_name(df: pd.DataFrame) -> str:
    if len(df.columns) == 0:
        raise ValueError("Excel has no columns")
    # Prefer an explicit 'SEC_ID' match (case-insensitive), else use first column
    for col in df.columns:
        if str(col).strip().lower() == "sec_id":
            return col
    return df.columns[0]


def build_row_maps(df: pd.DataFrame, id_col: str) -> Dict[str, int]:
    id_to_index: Dict[str, int] = {}
    for idx, val in df[id_col].items():
        key = str(val)
        if key not in id_to_index:
            id_to_index[key] = idx
    return id_to_index


def intersect_columns(df1: pd.DataFrame, df2: pd.DataFrame, exclude: Set[str]) -> List[str]:
    cols = [c for c in df1.columns if c in df2.columns and c not in exclude]
    return cols


def compare_rows(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col1: str,
    id_col2: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Set[str], Set[str]]:
    id_map1 = build_row_maps(df1, id_col1)
    id_map2 = build_row_maps(df2, id_col2)

    ids1 = set(id_map1.keys())
    ids2 = set(id_map2.keys())

    common_ids = ids1 & ids2
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1

    comparable_cols = intersect_columns(df1, df2, exclude={id_col1, id_col2})

    diffs1: Dict[str, List[str]] = {}
    diffs2: Dict[str, List[str]] = {}

    for sec_id in common_ids:
        row1 = df1.loc[id_map1[sec_id]]
        row2 = df2.loc[id_map2[sec_id]]
        diff_cols: List[str] = []
        for col in comparable_cols:
            v1 = row1[col]
            v2 = row2[col]
            if str(v1) != str(v2):
                diff_cols.append(col)
        if diff_cols:
            diffs1[sec_id] = diff_cols.copy()
            diffs2[sec_id] = diff_cols.copy()

    return diffs1, diffs2, only_in_1, only_in_2


def apply_highlights(
    src_path: str,
    out_path: str,
    sec_id_to_diff_cols: Dict[str, List[str]],
    sec_ids_only_in_this: Set[str],
    id_col_name: str,
):
    wb = load_workbook(src_path)
    ws = wb.active

    # Map headers to column index (1-based for openpyxl)
    header_row = 1
    headers = {}
    for col_idx, cell in enumerate(ws[header_row], start=1):
        headers[str(cell.value).strip() if cell.value is not None else ""] = col_idx

    # Build SEC_ID -> worksheet row number
    id_col_idx = headers.get(id_col_name)
    if id_col_idx is None:
        # Fallback to first column
        id_col_idx = 1
    sec_id_to_row: Dict[str, int] = {}
    for r in range(header_row + 1, ws.max_row + 1):
        val = ws.cell(row=r, column=id_col_idx).value
        key = str(val).strip() if val is not None else ""
        if key and key not in sec_id_to_row:
            sec_id_to_row[key] = r

    # Convenience header name to column index for diff columns
    diff_col_to_idx: Dict[str, int] = {}
    for col_name in ws.iter_rows(min_row=header_row, max_row=header_row, values_only=True):
        for idx, name in enumerate(col_name, start=1):
            if name is not None:
                diff_col_to_idx[str(name).strip()] = idx

    # 1) Highlight rows that have differences: whole row YELLOW, then specific cells RED
    for sec_id, diff_cols in sec_id_to_diff_cols.items():
        row_num = sec_id_to_row.get(sec_id)
        if not row_num:
            continue
        for c in range(1, ws.max_column + 1):
            ws.cell(row=row_num, column=c).fill = YELLOW_FILL
        for col_name in diff_cols:
            cidx = diff_col_to_idx.get(col_name)
            if cidx is not None:
                ws.cell(row=row_num, column=cidx).fill = RED_FILL

    # 2) Highlight rows only in this workbook: whole row BLUE
    for sec_id in sec_ids_only_in_this:
        row_num = sec_id_to_row.get(sec_id)
        if not row_num:
            continue
        for c in range(1, ws.max_column + 1):
            ws.cell(row=row_num, column=c).fill = BLUE_FILL

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)


def derive_output_paths(file1: str, file2: str, output_dir: str) -> Tuple[str, str]:
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]
    out1 = os.path.join(output_dir, f"{base1}__highlighted.xlsx")
    out2 = os.path.join(output_dir, f"{base2}__highlighted.xlsx")
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
            "Compare two Excel files by SEC_ID (first column or column named SEC_ID). "
            "Highlight differing cells red and full differing rows yellow. Rows with a SEC_ID only present in one file are highlighted blue in that file."
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

    if not os.path.exists(file1):
        print(f"Error: file not found: {file1}")
        return 1
    if not os.path.exists(file2):
        print(f"Error: file not found: {file2}")
        return 1

    try:
        df1 = read_excel_as_str(file1)
        df2 = read_excel_as_str(file2)
    except Exception as e:
        print(f"Failed to read Excel files: {e}")
        return 1

    id_col1 = get_sec_id_column_name(df1)
    id_col2 = get_sec_id_column_name(df2)

    diffs1, diffs2, only_in_1, only_in_2 = compare_rows(df1, df2, id_col1, id_col2)

    out1, out2 = derive_output_paths(file1, file2, args.output)

    try:
        apply_highlights(file1, out1, diffs1, only_in_1, id_col1)
        apply_highlights(file2, out2, diffs2, only_in_2, id_col2)
    except Exception as e:
        print(f"Failed to write highlighted workbooks: {e}")
        return 1

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
