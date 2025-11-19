"Module to restructure and condense ANCOVA regression results Excel files."

import os

from copy import copy
from pathlib import Path

import openpyxl

import numpy as np
import pandas as pd

from openpyxl.utils.dataframe import dataframe_to_rows

from utils.logging_config import get_logger

logger = get_logger(__name__)


FILE_PATH = Path(__file__).parents[2] / "results"


def restructure_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restructures a single sheet (DataFrame) from the ANCOVA regression results.

    The function consolidates scattered characteristic terms and treatment-by-characteristic
    interaction terms into single, consolidated rows, following a specific output format
    (e.g., the "Final savings" tab structure).

    The output structure order is:
    1. C(treatment)[T.Intervention X] + SE (for X=1, 2, 3)
    2. Characteristic + SE (consolidated main effect)
    3. C(treatment)[T.Intervention X]:Characteristic[T.1] + SE (consolidated interactions)
    4. Additional covariates (e.g., sreal_percent_pre) + SE
    5. Intercept + SE
    6. R-squared rows
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Lists to identify rows of interest
    interaction_rows = []
    characteristic_rows = []
    treatment_rows = []
    intercept_rows = []
    additional_covariate_rows = []
    r_squared_rows = []

    for idx, row in df.iterrows():
        row_label = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""

        if "C(treatment)[T.Intervention" in row_label:
            if "]:" in row_label:
                interaction_rows.append(idx)
            else:
                treatment_rows.append(idx)
        elif row_label == "Intercept":
            intercept_rows.append(idx)
        elif "R-squared" in row_label:
            r_squared_rows.append(idx)
        # Identify characteristic rows (main effect, not interaction, not treatment, etc.)
        elif (
            row_label not in ["Intercept", "R-squared", "R-squared Adj."]
            and "C(treatment)" not in row_label
            and "pre" not in row_label  # General check for pre-treatment measures
            and len(row_label.strip()) > 0
        ):
            # Check if this row has data in columns other than the first
            if any(pd.notna(row.iloc[i]) for i in range(1, len(row))):
                characteristic_rows.append(idx)
        # Identify additional covariate rows (pre-treatment measures)
        elif "pre" in row_label:
            if any(pd.notna(row.iloc[i]) for i in range(1, len(row))):
                additional_covariate_rows.append(idx)

    if not treatment_rows:
        return df  # No restructuring needed

    num_cols = len(df.columns)

    # --- 1. Extract and Consolidate Data ---

    # 1.1. Treatment Main Effects (already consolidated, just need to extract)
    treatment_1_coef = (
        df.iloc[treatment_rows[0]].tolist()
        if len(treatment_rows) > 0
        else [None] * num_cols
    )
    treatment_1_se = (
        df.iloc[treatment_rows[0] + 1].tolist()
        if len(treatment_rows) > 0 and treatment_rows[0] + 1 < len(df)
        else [None] * num_cols
    )

    treatment_2_coef = (
        df.iloc[treatment_rows[1]].tolist()
        if len(treatment_rows) > 1
        else [None] * num_cols
    )
    treatment_2_se = (
        df.iloc[treatment_rows[1] + 1].tolist()
        if len(treatment_rows) > 1 and treatment_rows[1] + 1 < len(df)
        else [None] * num_cols
    )

    treatment_3_coef = (
        df.iloc[treatment_rows[2]].tolist()
        if len(treatment_rows) > 2
        else [None] * num_cols
    )
    treatment_3_se = (
        df.iloc[treatment_rows[2] + 1].tolist()
        if len(treatment_rows) > 2 and treatment_rows[2] + 1 < len(df)
        else [None] * num_cols
    )

    # 1.2. Characteristic Main Effect (Consolidation)
    characteristic_coef = [None] * num_cols
    characteristic_se = [None] * num_cols
    characteristic_coef[0] = "Characteristic"

    for idx in characteristic_rows:
        row_data = df.iloc[idx]
        for col_idx in range(1, num_cols):
            if pd.notna(row_data.iloc[col_idx]):
                characteristic_coef[col_idx] = row_data.iloc[col_idx]
                if idx + 1 < len(df):
                    characteristic_se[col_idx] = df.iloc[idx + 1, col_idx]

    # 1.3. Interaction Effects (Consolidation)
    intervention_1_coef = [None] * num_cols
    intervention_1_se = [None] * num_cols
    intervention_2_coef = [None] * num_cols
    intervention_2_se = [None] * num_cols
    intervention_3_coef = [None] * num_cols
    intervention_3_se = [None] * num_cols

    intervention_1_coef[0] = "C(treatment)[T.Intervention 1]:Characteristic[T.1]"
    intervention_2_coef[0] = "C(treatment)[T.Intervention 2]:Characteristic[T.1]"
    intervention_3_coef[0] = "C(treatment)[T.Intervention 3]:Characteristic[T.1]"

    i = 0
    while i < len(df):
        row_data = df.iloc[i]
        row_label = str(row_data.iloc[0]) if pd.notna(row_data.iloc[0]) else ""

        if "C(treatment)[T.Intervention 1]:" in row_label:
            for col_idx in range(1, num_cols):
                if pd.notna(row_data.iloc[col_idx]):
                    intervention_1_coef[col_idx] = row_data.iloc[col_idx]
                    if i + 1 < len(df):
                        intervention_1_se[col_idx] = df.iloc[i + 1, col_idx]
            i += 2
            continue
        elif "C(treatment)[T.Intervention 2]:" in row_label:
            for col_idx in range(1, num_cols):
                if pd.notna(row_data.iloc[col_idx]):
                    intervention_2_coef[col_idx] = row_data.iloc[col_idx]
                    if i + 1 < len(df):
                        intervention_2_se[col_idx] = df.iloc[i + 1, col_idx]
            i += 2
            continue
        elif "C(treatment)[T.Intervention 3]:" in row_label:
            for col_idx in range(1, num_cols):
                if pd.notna(row_data.iloc[col_idx]):
                    intervention_3_coef[col_idx] = row_data.iloc[col_idx]
                    if i + 1 < len(df):
                        intervention_3_se[col_idx] = df.iloc[i + 1, col_idx]
            i += 2
            continue
        i += 1

    # 1.4. Pre-treatment Measure (already consolidated, just need to extract)
    additional_covariate_data = []
    for idx in additional_covariate_rows:
        additional_covariate_data.append(df.iloc[idx].tolist())
        if idx + 1 < len(df):
            additional_covariate_data.append(df.iloc[idx + 1].tolist())

    # 1.5. Intercept (already consolidated, just need to extract)
    intercept_coef = (
        df.iloc[intercept_rows[0]].tolist() if intercept_rows else [None] * num_cols
    )
    intercept_se = (
        df.iloc[intercept_rows[0] + 1].tolist()
        if intercept_rows and intercept_rows[0] + 1 < len(df)
        else [None] * num_cols
    )

    # 1.6. R-squared (already consolidated, just need to extract)
    r_squared_data = [df.iloc[idx].tolist() for idx in r_squared_rows]

    # --- 2. Assemble the Final DataFrame in the Specified Order ---

    final_rows = []

    # Header row
    final_rows.append(df.iloc[0].tolist())

    # 1. Intervention 1 + SE
    final_rows.append(treatment_1_coef)
    final_rows.append(treatment_1_se)

    # 2. Intervention 2 + SE
    final_rows.append(treatment_2_coef)
    final_rows.append(treatment_2_se)

    # 3. Intervention 3 + SE
    final_rows.append(treatment_3_coef)
    final_rows.append(treatment_3_se)

    # 4. Characteristic + SE
    final_rows.append(characteristic_coef)
    final_rows.append(characteristic_se)

    # 5. Int 1×Char + SE
    final_rows.append(intervention_1_coef)
    final_rows.append(intervention_1_se)

    # 6. Int 2×Char + SE
    final_rows.append(intervention_2_coef)
    final_rows.append(intervention_2_se)

    # 7. Int 3×Char + SE
    final_rows.append(intervention_3_coef)
    final_rows.append(intervention_3_se)

    # 8. Pre-treat measure + SE
    final_rows.extend(additional_covariate_data)

    # 9. Intercept + SE
    final_rows.append(intercept_coef)
    final_rows.append(intercept_se)

    # 10. R-squared
    final_rows.extend(r_squared_data)

    result_df = pd.DataFrame(final_rows, columns=df.columns)

    return result_df


def restructure_excel_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    Reads an Excel file, restructures all sheets using restructure_sheet,
    and saves the result to a new Excel file.

    Args:
        input_path: Path to the input Excel file.
        output_path: Path where the restructured Excel file will be saved.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Load the original workbook
    wb = openpyxl.load_workbook(input_path)

    # Create a new workbook for the output
    output_wb = openpyxl.Workbook()
    output_wb.remove(output_wb.active)  # Remove default sheet

    # Process each sheet
    for sheet_name in wb.sheetnames:
        print(f"Processing sheet: {sheet_name}")

        # Read the sheet
        df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)

        # Restructure the sheet
        restructured_df = restructure_sheet(df)

        # Create a new sheet in the output workbook
        new_sheet = output_wb.create_sheet(title=sheet_name)

        # Copy the restructured data
        for r_idx, row in enumerate(
            dataframe_to_rows(restructured_df, index=False, header=False), 1
        ):
            for c_idx, value in enumerate(row, 1):
                new_sheet.cell(row=r_idx, column=c_idx, value=value)

        # Copy formatting from original sheet (if possible)
        original_sheet = wb[sheet_name]
        for row in new_sheet.iter_rows():
            for cell in row:
                if (
                    cell.row <= original_sheet.max_row
                    and cell.column <= original_sheet.max_column
                ):
                    try:
                        original_cell = original_sheet.cell(
                            row=cell.row, column=cell.column
                        )
                        if original_cell.has_style:
                            cell.font = copy(original_cell.font)
                            cell.border = copy(original_cell.border)
                            cell.fill = copy(original_cell.fill)
                            cell.number_format = copy(original_cell.number_format)
                            cell.protection = copy(original_cell.protection)
                            cell.alignment = copy(original_cell.alignment)
                    except:
                        pass

    # Save the output file
    output_wb.save(output_path)
    print(f"Restructured file saved to: {output_path}")


if __name__ == "__main__":
    # Example usage (assuming the input file is in the current directory)
    input_file = FILE_PATH / "ancova_heterogeneous_effects_20251119-123200.xlsx"
    output_file = (
        FILE_PATH / "ancova_cognitivemeasures_restructured_20251119-123200.xlsx"
    )

    restructure_excel_file(input_file, output_file)
