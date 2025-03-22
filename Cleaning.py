import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os

def standardize_dtypes(df):
    for c in df.columns:
        if 'date' in c.lower():
            df[c] = pd.to_datetime(df[c], errors='coerce')
        elif df[c].dtype == 'object':
            df[c] = df[c].str.strip().str.lower().str.replace(r'[^\w\s]', '', regex=True)
        else:
            df[c] = pd.to_numeric(df[c], errors='ignore')
    return df

def create_year_quarter(df):
    if {'year','quarter'}.issubset(df.columns):
        df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    return df

def get_group_keys(df, defaults):
    possible = ['state','city','market','region','building_classification','year_quarter']
    keys = [k for k in possible if k in df.columns]
    return keys or defaults

def dynamic_clean_group(g):
    r = g.notnull().mean(axis=1)
    m = g.isnull().mean()
    year = g['year'].iloc[0] if 'year' in g.columns else None
    if year == 2018:
        row_thresh, col_thresh = 0.4, 0.6
    else:
        row_thresh = max(0.2, r.median() - 0.1)
        col_thresh = min(0.9, m.median() + 0.1)
    g = g[r >= row_thresh]
    drop_cols = m[m >= col_thresh].index
    g = g.drop(columns=drop_cols, errors='ignore')
    for c in g.columns:
        if pd.api.types.is_numeric_dtype(g[c]):
            g[c].fillna(g[c].median(), inplace=True)
        else:
            mode = g[c].mode()
            g[c].fillna(mode[0] if not mode.empty else '', inplace=True)
    return g

def remove_outliers(df):
    for c in df.select_dtypes(include='number').columns:
        q1, q3 = df[c].quantile([0.25,0.75])
        iqr = q3 - q1
        df = df[(df[c] >= q1 - 1.5*iqr) & (df[c] <= q3 + 1.5*iqr)]
    return df

def normalize(df):
    for c in df.select_dtypes(include='number').columns:
        mi, mx = df[c].min(), df[c].max()
        if mi != mx:
            df[c] = (df[c] - mi)/(mx - mi)
        else:
            df[c] = 0.0
    return df

def main():
    # Avoid re-running if CSVs already exist
    if all(os.path.exists(fn) for fn in [
        'cleaned_leases_full_dynamic.csv',
        'cleaned_occupancy_full_dynamic.csv',
        'cleaned_price_avail_full_dynamic.csv',
        'cleaned_unemployment_full_dynamic.csv'
    ]):
        print("Cleaned CSVs already exist. Delete them if you want to re-run cleaning.")
        return

    # Load + deduplicate
    leases      = pd.read_csv('Leases.csv').drop_duplicates()
    occupancy   = pd.read_csv('Major Market Occupancy Data.csv').drop_duplicates()
    price_avail = pd.read_csv('Price and Availability Data.csv').drop_duplicates()
    unemp       = pd.read_csv('Unemployment.csv').drop_duplicates()

    # Create year_quarter
    for df in (leases, occupancy, price_avail):
        create_year_quarter(df)

    # Build grouping keys
    keys = {
        'leases': get_group_keys(leases, ['market','year_quarter']),
        'occupancy': get_group_keys(occupancy, ['market','year_quarter']),
        'price_avail': get_group_keys(price_avail, ['market','year_quarter']),
        'unemployment': ['state','year'] if {'state','year'}.issubset(unemp.columns) else ['year']
    }

    # Clean
    cleaned_leases = normalize(remove_outliers(standardize_dtypes(
                           leases.groupby(keys['leases']).apply(dynamic_clean_group).reset_index(drop=True))))
    cleaned_occupancy = normalize(remove_outliers(standardize_dtypes(
                           occupancy.groupby(keys['occupancy']).apply(dynamic_clean_group).reset_index(drop=True))))
    cleaned_price_avail = normalize(remove_outliers(standardize_dtypes(
                           price_avail.groupby(keys['price_avail']).apply(dynamic_clean_group).reset_index(drop=True))))
    cleaned_unemployment = normalize(remove_outliers(standardize_dtypes(
                           unemp.groupby(keys['unemployment']).apply(dynamic_clean_group).reset_index(drop=True))))

    # Save
    cleaned_leases.to_csv('cleaned_leases_full_dynamic.csv', index=False)
    cleaned_occupancy.to_csv('cleaned_occupancy_full_dynamic.csv', index=False)
    cleaned_price_avail.to_csv('cleaned_price_avail_full_dynamic.csv', index=False)
    cleaned_unemployment.to_csv('cleaned_unemployment_full_dynamic.csv', index=False)
    print("Cleaning complete. CSVs saved.")

if __name__ == "__main__":
    main()
