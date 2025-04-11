import pandas as pd
import numpy as np

def parse_date_columns(df):
    """
    For any column whose name contains 'date' or 'signed',
    attempt to parse it as datetime. E.g. 'lease_date', 'monthsigned'.
    """
    # Example pattern: if 'date' or 'signed' is in the column name
    date_like_cols = [c for c in df.columns if any(k in c.lower() for k in ['date','signed'])]
    for col in date_like_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def fill_year_qtr_month_from_date(df):
    """
    If we have a parsed date column, fill missing year/quarter/month from that date if possible.
    - If multiple date columns exist, we just pick the first non-null date for each row.
    - If year/quarter/month are still missing after that, fill with 0 to keep them integer.
    """
    # Identify date columns (already parsed as datetime64)
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    if not dt_cols:
        return df  # No parsed date columns => do nothing

    # We'll attempt each row with the first valid date in any dt_col
    for idx in df.index:
        # Already have year? If not, try fill from a date
        row_has_year = ('year' in df.columns and pd.notnull(df.at[idx, 'year']))
        row_has_quarter = ('quarter' in df.columns and pd.notnull(df.at[idx, 'quarter']))
        row_has_month = ('month' in df.columns and pd.notnull(df.at[idx, 'month']))

        if row_has_year and row_has_quarter and row_has_month:
            continue  # Already have them all

        # Find a date column with a valid datetime
        date_val = None
        for c in dt_cols:
            val = df.at[idx, c]
            if pd.notnull(val):
                date_val = val
                break

        if pd.isnull(date_val):
            continue  # no valid date => can't fill from date

        # Fill missing year/quarter/month from date_val
        if 'year' in df.columns and not row_has_year:
            df.at[idx, 'year'] = date_val.year
        if 'month' in df.columns and not row_has_month:
            df.at[idx, 'month'] = date_val.month
        if 'quarter' in df.columns and not row_has_quarter:
            # quarter is in [1..4]
            df.at[idx, 'quarter'] = date_val.quarter

    # Finally, if year/quarter/month are still missing => fill with 0
    for col in ['year','quarter','month']:
        if col in df.columns:
            df[col].fillna(0, inplace=True)  # keep them integer
            df[col] = df[col].astype(int, errors='ignore')

    return df

def minimal_clean_with_dates(input_csv, output_csv):
    """
    1. Load CSV (no special date parse in read_csv).
    2. Remove exact duplicates.
    3. Parse date-like columns (names contain 'date' or 'signed').
    4. If year/quarter/month exist, fill missing from date columns if possible, else 0.
    5. Fill missing numeric columns (besides year/quarter/month) with median.
    6. Fill missing object columns with 'missing'.
    7. Save to new CSV, preserving column names.

    This is a moderate approach: it modifies data by filling missing values
    but does not rename columns or drop them, preserving the structure.
    """

    # --- 1) Load CSV ---
    df = pd.read_csv(input_csv)
    print(f"\nLoaded '{input_csv}' with shape={df.shape}")

    # --- 2) Remove exact duplicates ---
    df.drop_duplicates(inplace=True)
    print(f"After dropping duplicates: shape={df.shape}")

    # --- 3) Parse date-like columns ---
    df = parse_date_columns(df)

    # --- 4) Fill year/quarter/month from date columns ---
    df = fill_year_qtr_month_from_date(df)

    # --- 5) Fill missing numeric columns with median (excluding year/quarter/month) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    protected_time_cols = {'year','quarter','month'}
    for col in numeric_cols:
        if col.lower() not in protected_time_cols:
            median_val = df[col].median(skipna=True)
            df[col].fillna(median_val, inplace=True)

    # --- 6) Fill missing object/string columns with 'missing' ---
    object_cols = df.select_dtypes(include=[object]).columns
    for col in object_cols:
        df[col].fillna('missing', inplace=True)

    # Also ensure year/quarter/month remain int
    for col in ['year','quarter','month']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore', downcast='integer')

    # --- 7) Save to new CSV ---
    df.to_csv(output_csv, index=False)
    print(f"Saved minimal-clean file to '{output_csv}' with shape={df.shape}")

# ------------------------------------------------------------------------
# Example usage for your 4 original CSV files
# ------------------------------------------------------------------------

if __name__ == "__main__":
    minimal_clean_with_dates("Leases.csv", "Leases_minimalclean.csv")
    minimal_clean_with_dates("Major Market Occupancy Data.csv", "Occupancy_minimalclean.csv")
    minimal_clean_with_dates("Price and Availability Data.csv", "PriceAvail_minimalclean.csv")
    minimal_clean_with_dates("Unemployment.csv", "Unemployment_minimalclean.csv")

    print("\nAll done. Each file has missing numeric values filled with medians, "
          "string columns filled with 'missing', date columns parsed, and "
          "year/quarter/month preserved as integers.")
