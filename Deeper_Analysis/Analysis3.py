import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1) LOAD THE MINIMAL-CLEAN DATASETS
# ---------------------------------------------------------------------
leases = pd.read_csv("Leases_minimalclean.csv")
occupancy = pd.read_csv("Occupancy_minimalclean.csv")
price_avail = pd.read_csv("PriceAvail_minimalclean.csv")
unemp = pd.read_csv("Unemployment_minimalclean.csv")

print("Loaded minimal-clean CSVs.")
print("leases.shape =", leases.shape)
print("occupancy.shape =", occupancy.shape)
print("price_avail.shape =", price_avail.shape)
print("unemp.shape =", unemp.shape)

# Inspect column names (optional)
print("\nLEASES columns:", leases.columns.tolist())
print("OCCUPANCY columns:", occupancy.columns.tolist())
print("PRICE_AVAIL columns:", price_avail.columns.tolist())
print("UNEMP columns:", unemp.columns.tolist())

# ---------------------------------------------------------------------
# 2) MERGE INTO ONE MASTER DATAFRAME
#    We'll merge on (market,year,quarter) for occupancy/price_avail,
#    and on (state,year) for unemployment, as an example.
# ---------------------------------------------------------------------

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
print("Columns:", df_merged.columns.tolist())

# Convert leasedSF to numeric (just to be safe)
df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")


# Define "small" vs. "large" lease categories
df_merged["size_category"] = df_merged["leasedSF"].apply(
    lambda sf: "Large (>=10k)" if sf >= 10000 else "Small (<10k)"
)


# Group by (year, size_category) => sum of leasedSF
groupA = df_merged.groupby(["year","size_category"])["leasedSF"].sum().reset_index()

# Pivot so columns are size_category => stacked
pivotA = groupA.pivot(index="year", columns="size_category", values="leasedSF").fillna(0)

# Stacked bar
pivotA.plot(kind="bar", stacked=True, figsize=(9,5))
plt.title("Total LeasedSF by Year, Stacked: Small vs. Large")
plt.xlabel("Year")
plt.ylabel("LeasedSF")
plt.xticks(rotation=0)
plt.legend(title="Size Category")
plt.tight_layout()
plt.show()

"""
STORY / INSIGHT:
- We can see which years had more small vs. large leases in total.
- If we see that large leases dominate certain years (e.g., 2019) but not others,
  that might reflect macro trends or big corporate relocations.
"""


if "region" in df_merged.columns and "internal_class" in df_merged.columns:
    # Summarize count of leases by (region, internal_class, year)
    groupB = df_merged.groupby(["year","region","internal_class"])["leasedSF"].count().reset_index(name="count_leases")

    # Focus on a single year or do a multi-year grouped bar. Let's do multi-year for demonstration.
    # For a simpler chart, let's filter to a single year or to top 4 regions, etc.
    # Example: top 4 regions by total row counts:
    top_regions = (groupB["region"].value_counts().head(4).index)
    groupB = groupB[groupB["region"].isin(top_regions)]

    # Pivot so columns are internal_class, rows are (year,region)
    pivotB = groupB.pivot_table(index=["year","region"], columns="internal_class", values="count_leases", fill_value=0)
    pivotB = pivotB.reset_index()
    
    # We'll create a grouped bar chart for each region within each year
    # Easiest approach: loop over years, or we can do a stacked approach. We'll do stacked for brevity:
    # First pivot again so index=year,region, columns=class, values => stacked
    # Then we might produce multiple subplots or just one big stacked chart.

    pivotB.set_index(["year","region"], inplace=True)
    pivotB.plot(kind="bar", stacked=True, figsize=(12,5))
    plt.title("Count of Leases by Region & Building Class Over Time (top 4 regions)")
    plt.xlabel("Year, Region")
    plt.ylabel("Number of Lease Records")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Building Class")
    plt.tight_layout()
    plt.show()

    """
    STORY / INSIGHT:
    - This reveals if certain regions prefer Class A or Class B or O, etc.
    - If the 'region' is, say, "Northeast," we might see more Class A in recent years.
    - It helps clients see which regions have a heavier tilt towards higher-end or lower-end buildings.
    """


if "region" in df_merged.columns and "internal_industry" in df_merged.columns:
    # Summarize total leasedSF by (region, internal_industry)
    groupC = df_merged.groupby(["region","internal_industry"])["leasedSF"].sum().reset_index()

    # We'll take top 5 industries overall for clarity
    top5_industries = (groupC.groupby("internal_industry")["leasedSF"].sum().nlargest(5).index)
    groupC = groupC[groupC["internal_industry"].isin(top5_industries)]

    # Pivot => rows=region, columns=industry, values=leasedSF => stacked bar
    pivotC = groupC.pivot(index="region", columns="internal_industry", values="leasedSF").fillna(0)
    # For a simpler chart, maybe keep top 6 or 7 regions as well
    top_regions = pivotC.sum(axis=1).nlargest(7).index
    pivotC = pivotC.loc[top_regions]

    pivotC.plot(kind="bar", stacked=True, figsize=(10,5))
    plt.title("Total LeasedSF by Region, Stacked by Top 5 Industries")
    plt.ylabel("LeasedSF")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Industry")
    plt.tight_layout()
    plt.show()

    """
    STORY / INSIGHT:
    - We see which region is dominated by which industries. For example, 'Region=West' might be heavy
      in 'Tech', whereas 'Region=South' might be heavier in 'Finance' or 'Healthcare'.
    - This helps a prospective tenant see if their sector is clustering in certain regions.
    """
