import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ------------------------------------------------------------
    # 1) LOAD MINIMAL-CLEAN DATASETS
    # ------------------------------------------------------------
    leases      = pd.read_csv("Leases_minimalclean.csv")
    occupancy   = pd.read_csv("Occupancy_minimalclean.csv")
    price_avail = pd.read_csv("PriceAvail_minimalclean.csv")
    unemp       = pd.read_csv("Unemployment_minimalclean.csv")

    print("Loaded minimal-clean CSVs.")
    print("leases.shape =", leases.shape)
    print("occupancy.shape =", occupancy.shape)
    print("price_avail.shape =", price_avail.shape)
    print("unemp.shape =", unemp.shape)

    # ------------------------------------------------------------
    # 2) MERGE INTO ONE MASTER DATAFRAME
    #    We'll merge on (market, year, quarter) for occupancy/price_avail,
    #    and on (state, year) for unemployment, as an example.
    # ------------------------------------------------------------

    # Merge occupancy into leases
    cols_occ = ["market", "year", "quarter", "occupancy_proportion"]
    df_merged = pd.merge(
        leases,
        occupancy[cols_occ],
        on=["market", "year", "quarter"],
        how="left"
    )

    # Merge price_avail (which has 'overall_rent')
    cols_price = ["market", "year", "quarter", "overall_rent"]
    df_merged = pd.merge(
        df_merged,
        price_avail[cols_price],
        on=["market", "year", "quarter"],
        how="left"
    )

    # Merge unemployment on (state, year)
    cols_unemp = ["state", "year", "unemployment_rate"]
    df_merged = pd.merge(
        df_merged,
        unemp[cols_unemp],
        on=["state","year"],
        how="left"
    )

    print("\nAfter merging all data:")
    print("df_merged.shape =", df_merged.shape)
    print("Columns:", df_merged.columns.tolist())

    # ------------------------------------------------------------
    # 3) HANDLE ANY COLUMN-NAME COLLISIONS
    #    If your final df_merged has 'overall_rent_x' and 'overall_rent_y'
    #    instead of a single 'overall_rent', unify them.
    # ------------------------------------------------------------
    # If you see 'overall_rent_x'/'overall_rent_y' in df_merged.columns, do:
    if "overall_rent_x" in df_merged.columns or "overall_rent_y" in df_merged.columns:
        # Use fillna to combine them:
        df_merged["overall_rent"] = (
            df_merged.get("overall_rent_x")    # returns None if not present
            .fillna(df_merged.get("overall_rent_y")) 
            if "overall_rent_x" in df_merged.columns else df_merged.get("overall_rent_y")
        )
        # Drop the old columns to avoid confusion
        drop_cols = [c for c in ["overall_rent_x","overall_rent_y"] if c in df_merged.columns]
        df_merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Convert leasedSF to numeric, just to be safe
    df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")

    # Quick check:
    print("\nColumns after unifying 'overall_rent':", df_merged.columns.tolist())

    # ------------------------------------------------------------
    # 4) YEAR-OVER-YEAR (YOY) ANALYSIS BY REGION
    #    We'll compute average rent & occupancy by (region, year),
    #    then measure yoy changes.
    # ------------------------------------------------------------
    group_region = df_merged.groupby(["region","year"]).agg({
        "overall_rent": "mean",
        "occupancy_proportion": "mean"
    }).reset_index()

    # Sort by region & year so we can create lag columns
    group_region.sort_values(["region","year"], inplace=True)

    # Create lag columns
    group_region["rent_lag"] = group_region.groupby("region")["overall_rent"].shift(1)
    group_region["occ_lag"]  = group_region.groupby("region")["occupancy_proportion"].shift(1)

    # YOY % changes
    group_region["rent_yoy_pct"] = (
        (group_region["overall_rent"] - group_region["rent_lag"]) 
        / group_region["rent_lag"] * 100
    )
    group_region["occ_yoy_pct"] = (
        (group_region["occupancy_proportion"] - group_region["occ_lag"]) 
        / group_region["occ_lag"] * 100
    )

    # ------------------------------------------------------------
    # 5) PLOT LATEST YEAR'S YOY RENT CHANGES
    # ------------------------------------------------------------
    latest_year = group_region["year"].max()
    latest_mask = (group_region["year"] == latest_year) & (group_region["rent_yoy_pct"].notnull())
    latest_rent = group_region.loc[latest_mask, ["region","rent_yoy_pct"]]
    latest_rent.sort_values("rent_yoy_pct", ascending=False, inplace=True)

    plt.figure(figsize=(8,4))
    plt.bar(latest_rent["region"], latest_rent["rent_yoy_pct"], color="orange")
    plt.title(f"YoY % Change in Overall Rent by Region ({latest_year} vs. {latest_year-1})")
    plt.ylabel("YoY % Rent Growth")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.show()

    # OPTIONAL: Similarly, you could plot yoy changes in occupancy or do more advanced analysis.

    print("\nDONE! You have now performed a region-based YoY analysis on average rent.")

if __name__ == "__main__":
    main()
