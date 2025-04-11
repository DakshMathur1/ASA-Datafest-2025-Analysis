import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########################################################
# 1) LOAD THE MINIMAL-CLEAN FILES
########################################################
leases = pd.read_csv("Leases_minimalclean.csv")
occupancy = pd.read_csv("Occupancy_minimalclean.csv")
price_avail = pd.read_csv("PriceAvail_minimalclean.csv")
unemp = pd.read_csv("Unemployment_minimalclean.csv")

print("Loaded minimal-clean files.")
print("leases:", leases.shape, 
      "occupancy:", occupancy.shape,
      "price_avail:", price_avail.shape, 
      "unemp:", unemp.shape)

########################################################
# 2) MERGE DATASETS
#    - Occupancy => merges on (market,year,quarter)
#    - PriceAvail => merges on (market,year,quarter)
#    - Unemployment => merges on (state,year)
########################################################
cols_occ = ["market","year","quarter","occupancy_proportion"]
df_merged = pd.merge(
    leases,
    occupancy[cols_occ],
    on=["market","year","quarter"],
    how="left"
)

cols_price = ["market","year","quarter","overall_rent"]
df_merged = pd.merge(
    df_merged,
    price_avail[cols_price],
    on=["market","year","quarter"],
    how="left"
)

cols_unemp = ["state","year","unemployment_rate"]
df_merged = pd.merge(
    df_merged,
    unemp[cols_unemp],
    on=["state","year"],
    how="left"
)

print("\nAfter merging all data:")
print("df_merged.shape =", df_merged.shape)
print("Columns in df_merged:", df_merged.columns.tolist())

# Convert leasedSF to numeric
df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")

# If you ended up with overall_rent_x / overall_rent_y, unify them:
if "overall_rent_x" in df_merged.columns or "overall_rent_y" in df_merged.columns:
    df_merged["overall_rent_merged"] = df_merged.get("overall_rent_x").combine_first(df_merged.get("overall_rent_y"))
    df_merged["overall_rent"] = df_merged["overall_rent_merged"]
    # You can drop the old columns if you want:
    drop_cols = [c for c in ["overall_rent_x","overall_rent_y","overall_rent_merged"] if c in df_merged.columns]
    df_merged.drop(columns=drop_cols, inplace=True, errors="ignore")

print("\nSample of merged data:\n", df_merged.head(5))

########################################################
# 3) SUBMARKET vs. INDUSTRY vs. YEAR PIVOT
#    - Summarize total leasedSF
#    - Make a heatmap if year is truly a pivot dimension
########################################################
if "internal_submarket" in df_merged.columns and "internal_industry" in df_merged.columns:
    pivot_si = (
        df_merged
        .groupby(["year","internal_submarket","internal_industry"])["leasedSF"]
        .sum()
        .reset_index()
        .pivot_table(
            index=["internal_submarket","internal_industry"],
            columns="year",
            values="leasedSF",
            fill_value=0
        )
    )
    print("\n--- Submarket vs. Industry vs. Year pivot (leasedSF) ---")
    print(pivot_si.head(10))

    # Try to pick a single year from the pivot columns to do a heatmap
    # We'll pick 2024 as an example
    year_pick = 2024
    if year_pick in pivot_si.columns:
        pivot_yr = pivot_si[[year_pick]]  # single-column DataFrame
        # rename the column so it's not just 2024
        pivot_yr.columns = ["leasedSF_{}".format(year_pick)]

        # The pivot_yr has a (submarket, industry) index. For a heatmap, we might do:
        #  - re-index or convert to wide shape. 
        # But it's 2D index. We'll do a simpler approach: unstack the submarket or industry.

        # Unstack to get submarket as rows, industry as columns (or vice versa)
        pivot_heat = pivot_yr.unstack(level="internal_industry", fill_value=0)
        # This produces a multi-level column. We'll do a simple .values approach or a custom plot.

        # We'll just do a quick seaborn heatmap
        plt.figure(figsize=(10,6))
        sns.heatmap(
            pivot_heat.values, 
            cmap="Blues", 
            xticklabels=pivot_heat.columns.levels[-1],  # the industries
            yticklabels=pivot_heat.index,              # the submarkets
            square=False
        )
        plt.title(f"Heatmap of LeasedSF in {year_pick}\n(Rows=submarket, Cols=industry)")
        plt.tight_layout()
        plt.show()

########################################################
# 4) CORRELATION HEATMAP AMONG SELECT COLUMNS
#    (occupancy, overall_rent, unemployment_rate, leasedSF, etc.)
########################################################
# We'll pick a set of numeric columns to see correlation
num_cols = ["leasedSF","overall_rent","occupancy_proportion","unemployment_rate"]
num_cols = [c for c in num_cols if c in df_merged.columns]
if len(num_cols) > 2:
    subset_corr = df_merged[num_cols].dropna()
    corr_matrix = subset_corr.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)
    plt.title("Correlation Heatmap: Key Numeric Columns")
    plt.show()

########################################################
# 5) YEAR-OVER-YEAR (YoY) CHANGES BY REGION
#    - Summarize average rent, total leasedSF, etc. by region+year
#    - Then compute yoy changes
########################################################
if "region" in df_merged.columns:
    yoy = (
        df_merged
        .groupby(["region","year"])
        .agg({
            "leasedSF":"sum",
            "overall_rent":"mean"
        })
        .reset_index()
        .sort_values(["region","year"])
    )
    # SHIFT for yoy
    yoy["leasedSF_lag"] = yoy.groupby("region")["leasedSF"].shift(1)
    yoy["leasedSF_yoy"] = yoy["leasedSF"] - yoy["leasedSF_lag"]
    yoy["leasedSF_yoy_pct"] = (yoy["leasedSF_yoy"] / yoy["leasedSF_lag"])*100

    yoy["rent_lag"] = yoy.groupby("region")["overall_rent"].shift(1)
    yoy["rent_yoy"] = yoy["overall_rent"] - yoy["rent_lag"]
    yoy["rent_yoy_pct"] = (yoy["rent_yoy"] / yoy["rent_lag"])*100

    # Plot yoy% for latest year
    latest_year = yoy["year"].max()
    yoy_latest = yoy[yoy["year"]==latest_year].dropna(subset=["leasedSF_yoy_pct"])
    yoy_latest = yoy_latest.sort_values("leasedSF_yoy_pct", ascending=False)

    plt.figure(figsize=(8,4))
    plt.bar(yoy_latest["region"], yoy_latest["leasedSF_yoy_pct"], color="green")
    plt.title(f"YoY % Growth in LeasedSF by Region ({latest_year})")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("YoY Growth (%)")
    plt.tight_layout()
    plt.show()

########################################################
# 6) LOCATION QUOTIENT (LQ) for a sector, e.g. "tech"
########################################################
if "internal_industry" in df_merged.columns:
    # Overall sum
    total_leased = df_merged["leasedSF"].sum()
    tech_leased = df_merged.loc[df_merged["internal_industry"]=="tech","leasedSF"].sum()
    if total_leased > 0:
        national_tech_share = tech_leased / total_leased
    else:
        national_tech_share = 0

    city_agg = (
        df_merged.groupby("market")["leasedSF"]
        .sum()
        .reset_index(name="city_total_leased")
    )
    city_tech = (
        df_merged[df_merged["internal_industry"]=="tech"]
        .groupby("market")["leasedSF"]
        .sum()
        .reset_index(name="city_tech_leased")
    )
    city_lq = pd.merge(city_agg, city_tech, on="market", how="left").fillna({"city_tech_leased":0})
    city_lq["city_tech_share"] = city_lq["city_tech_leased"] / city_lq["city_total_leased"]
    city_lq["tech_LQ"] = city_lq["city_tech_share"] / national_tech_share

    city_lq = city_lq.sort_values("tech_LQ", ascending=False)
    print("\n--- Top 10 markets by Tech LQ ---")
    print(city_lq.head(10))

    # Quick bar plot
    top10_lq = city_lq.head(10)
    plt.figure(figsize=(9,5))
    plt.bar(top10_lq["market"], top10_lq["tech_LQ"], color="orange")
    plt.title("Top 10 Markets by Tech LQ")
    plt.ylabel("Tech Location Quotient")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

print("\nALL DONE! This script merges your minimal-clean data, unifies 'overall_rent', and performs a variety of advanced/fancy analyses/plots (no regression).")
