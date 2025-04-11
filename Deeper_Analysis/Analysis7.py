import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
# 1) LOAD AND MERGE MINIMAL-CLEAN DATA
###############################################################################
leases = pd.read_csv("Leases_minimalclean.csv")
occupancy = pd.read_csv("Occupancy_minimalclean.csv")
price_avail = pd.read_csv("PriceAvail_minimalclean.csv")
unemp = pd.read_csv("Unemployment_minimalclean.csv")

print("Loaded minimal-clean files.")
print("leases:", leases.shape, "occupancy:", occupancy.shape,
      "price_avail:", price_avail.shape, "unemp:", unemp.shape)

# Merge occupancy -> on (market, year, quarter)
cols_occ = ["market","year","quarter","occupancy_proportion"]
df_merged = pd.merge(
    leases,
    occupancy[cols_occ],
    on=["market","year","quarter"],
    how="left"
)

# Merge price_avail -> on (market, year, quarter)
cols_price = ["market","year","quarter","overall_rent"]
df_merged = pd.merge(
    df_merged,
    price_avail[cols_price],
    on=["market","year","quarter"],
    how="left"
)

# Merge unemp -> on (state, year)
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

# Convert leasedSF to numeric, just in case
df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")

###############################################################################
# 3) STACKED AREA CHART: Summarize LeasedSF by Industry Over Time
###############################################################################
# We assume columns: year, quarter, internal_industry, leasedSF
if all(c in df_merged.columns for c in ["year","quarter","internal_industry","leasedSF"]):
    df_ind = df_merged.groupby(["year","quarter","internal_industry"], dropna=False)["leasedSF"].sum().reset_index()
    df_ind["year_qtr"] = df_ind["year"].astype(str)+"Q"+df_ind["quarter"].astype(str)
    
    pivot_ind = df_ind.pivot(index="year_qtr", columns="internal_industry", values="leasedSF").fillna(0)
    pivot_ind = pivot_ind.sort_index()  # ensure time is sorted
    
    plt.figure(figsize=(10,5))
    pivot_ind.plot.area(stacked=True, alpha=0.8)
    plt.title("Stacked Area: LeasedSF by Industry Over Time")
    plt.xlabel("Year+Quarter")
    plt.ylabel("LeasedSF")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\n[Skipping stacked area chart: missing year/quarter/internal_industry/leasedSF]")

###############################################################################
# 4) BOXPLOT: Distribution of lease sizes by region
###############################################################################
# Using Seaborn for a nicer boxplot. 
if all(c in df_merged.columns for c in ["region","leasedSF"]):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,5))
    # Optionally filter out extreme outliers
    # e.g. df_merged[df_merged["leasedSF"] < 5000000]
    df_box = df_merged.copy()
    
    sns.boxplot(x="region", y="leasedSF", data=df_box, showfliers=False)
    plt.title("Distribution of Lease Sizes by Region (No Outliers Shown)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("\n[Skipping boxplot: missing region/leasedSF columns]")

###############################################################################
# 5) BUBBLE CHART: Occupancy vs Weighted Rent vs LeasedSF
###############################################################################
# Summarize at (market, year), computing occupancy, total leasedSF, weighted rent
if all(c in df_merged.columns for c in ["market","year","leasedSF","occupancy_proportion","overall_rent"]):
    # Weighted rent approach
    df_merged["weighted_rent"] = df_merged["overall_rent"] * df_merged["leasedSF"]
    
    group_bubble = df_merged.groupby(["market","year"], dropna=False).agg({
        "occupancy_proportion":"mean",
        "leasedSF":"sum",
        "weighted_rent":"sum"
    }).reset_index()
    
    group_bubble["wgt_rent"] = group_bubble["weighted_rent"] / group_bubble["leasedSF"].replace(0, np.nan)
    
    plt.figure(figsize=(9,5))
    plt.scatter(
        group_bubble["occupancy_proportion"], 
        group_bubble["wgt_rent"], 
        s=group_bubble["leasedSF"] / 10000,  # scale bubble size
        alpha=0.4, 
        c=group_bubble["year"],  # color by year
        cmap="viridis"
    )
    plt.colorbar(label="Year")
    plt.xlabel("Avg Occupancy Proportion")
    plt.ylabel("Weighted Rent")
    plt.title("Bubble Chart: Occupancy vs Weighted Rent vs LeasedSF Size (colored by Year)")
    plt.tight_layout()
    plt.show()
else:
    print("\n[Skipping bubble chart: missing occupancy_proportion/overall_rent/leasedSF or year]")

###############################################################################
# 6) (OPTIONAL) CHOROPLETH MAP: e.g., State-level leasing
###############################################################################
# Pseudocode only: requires a shapefile or lat/lon
"""
import geopandas as gpd
# states_gdf = gpd.read_file("path/to/states_shapefile.shp")
# Summarize df_merged by state => total leasedSF
df_state = df_merged.groupby("state")["leasedSF"].sum().reset_index()
# states_gdf = states_gdf.merge(df_state, on="state", how="left")
# states_gdf.plot(column="leasedSF", legend=True, cmap="OrRd", missing_kwds={"color":"lightgrey"})
plt.show()
"""

###############################################################################
# 7) ANY ADDITIONAL TRENDS WE MISSED?
###############################################################################
# Example: yoy changes by submarket & year
if all(c in df_merged.columns for c in ["internal_submarket","year","leasedSF"]):
    yoy_sub = df_merged.groupby(["internal_submarket","year"], dropna=False)["leasedSF"].sum().reset_index()
    yoy_sub = yoy_sub.sort_values(["internal_submarket","year"])
    yoy_sub["lag_sf"] = yoy_sub.groupby("internal_submarket")["leasedSF"].shift(1)
    yoy_sub["yoy_pct"] = ((yoy_sub["leasedSF"] - yoy_sub["lag_sf"])/yoy_sub["lag_sf"])*100
    
    # For the latest year, show top yoy% growth submarkets
    latest_year = yoy_sub["year"].max()
    yoy_latest = yoy_sub[yoy_sub["year"]==latest_year].dropna(subset=["yoy_pct"])
    yoy_latest = yoy_latest.sort_values("yoy_pct", ascending=False).head(10)
    
    plt.figure(figsize=(8,4))
    plt.barh(yoy_latest["internal_submarket"], yoy_latest["yoy_pct"], color="darkgreen")
    plt.title(f"Top 10 Submarkets by YoY Growth in {latest_year}")
    plt.xlabel("YoY % Growth in LeasedSF")
    plt.gca().invert_yaxis()  # so highest is on top
    plt.tight_layout()
    plt.show()
    
    print("\nSubmarkets with highest yoy% growth:")
    print(yoy_latest[["internal_submarket","yoy_pct"]])
else:
    print("\n[Skipping yoy submarket: missing internal_submarket/year/leasedSF]")

print("\nAll done! Above code merges your minimal-clean data and provides a variety of advanced plots (treemap, stacked area, boxplot, bubble chart, yoy submarket, etc.).")
