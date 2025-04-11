import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1) LOAD THE MINIMAL-CLEAN DATASETS
#    (These were produced by your minimal_clean_with_dates script.)
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

# OPTIONAL: Inspect column names
print("\nLEASES columns:", leases.columns.tolist())
print("OCCUPANCY columns:", occupancy.columns.tolist())
print("PRICE_AVAIL columns:", price_avail.columns.tolist())
print("UNEMP columns:", unemp.columns.tolist())

# ---------------------------------------------------------------------
# 2) MERGE INTO ONE MASTER DATAFRAME
#    We'll do step-by-step merges:
#    - Occupancy => merges on (market, year, quarter) if those exist
#    - PriceAvail => merges on (market, year, quarter)
#    - Unemployment => merges on (state, year)
#
# NOTE: Adjust the "on=..." keys as needed if your minimal-clean CSVs
#       have a 'year_quarter' column instead, or if they store 'quarter'
#       differently. The code below assumes each file has year/quarter
#       columns plus 'market' or 'state' consistently.
# ---------------------------------------------------------------------

# Step A: Merge occupancy with leases
# We'll pick the columns from occupancy that we need, e.g. occupancy_proportion
# or avg_occupancy_proportion (depending on your data).
cols_occ = ["market", "year", "quarter", "occupancy_proportion"]
df_merged = pd.merge(
    leases,
    occupancy[cols_occ],
    on=["market", "year", "quarter"],
    how="left"  # left-join: keep all leases rows
)

# Step B: Merge price_avail (for overall_rent or asking_rent)
cols_price = ["market", "year", "quarter", "overall_rent"]  # adjust if your col is different
df_merged = pd.merge(
    df_merged,
    price_avail[cols_price],
    on=["market", "year", "quarter"],
    how="left"
)

# Step C: Merge unemployment on (state, year)
# We assume unemp has columns like [state, year, unemployment_rate].
cols_unemp = ["state", "year", "unemployment_rate"]
df_merged = pd.merge(
    df_merged,
    unemp[cols_unemp],
    on=["state", "year"],
    how="left"
)

print("\nAfter merging all data:")
print("df_merged.shape =", df_merged.shape)
print("df_merged columns:", df_merged.columns.tolist())

# ---------------------------------------------------------------------
# 3) FOCUS ON LARGE LEASES (≥ 10,000 SF)
#    The column for lease size might be 'leasedSF' or something else.
# ---------------------------------------------------------------------
df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")
large_leases = df_merged[df_merged["leasedSF"] >= 10000].copy()
print("\nNumber of large leases (≥10k SF):", len(large_leases))

# ---------------------------------------------------------------------
# 4) EXAMPLE ANALYSES
# ---------------------------------------------------------------------

# =========== (A) Basic Time Trend of Large-Lease Activity ============
# Summarize total leasedSF by year+quarter across entire dataset
group_a = large_leases.groupby(["year", "quarter"])["leasedSF"].sum().reset_index()
group_a["year_quarter"] = group_a["year"].astype(str) + "Q" + group_a["quarter"].astype(str)

# Plot
plt.figure(figsize=(10,5))
plt.bar(group_a["year_quarter"], group_a["leasedSF"])
plt.title("Total LeasedSF (≥10k) by Year+Quarter")
plt.xticks(rotation=45)
plt.ylabel("LeasedSF")
plt.tight_layout()
plt.show()

# COMMENTARY:
# This gives a high-level view of how total large-lease volume changes over time.
# You might see a drop in certain quarters (e.g. 2020 Q2 due to Covid?), or
# a rebound in 2021 or 2022, etc.

# =========== (B) Occupancy vs. LeasedSF Over Time in a Major Market ============
# Suppose we pick "Manhattan" or "San Francisco" or any major market
major_market = "manhattan"  # adjust as you like

mm_data = large_leases[large_leases["market"].str.lower() == major_market]
group_b = mm_data.groupby(["year", "quarter"]).agg({
    "leasedSF": "sum",
    "occupancy_proportion": "mean"
}).reset_index()

group_b["year_quarter"] = group_b["year"].astype(str) + "Q" + group_b["quarter"].astype(str)

# Plot on dual y-axis: bar for leasedSF, line for occupancy_proportion
fig, ax1 = plt.subplots(figsize=(9,5))
ax2 = ax1.twinx()

ax1.bar(group_b["year_quarter"], group_b["leasedSF"], color="skyblue", label="LeasedSF")
ax2.plot(group_b["year_quarter"], group_b["occupancy_proportion"], color="red", marker="o", label="Occupancy %")

ax1.set_title(f"Large Leases in {major_market.title()}: LeasedSF vs. Occupancy")
ax1.set_xlabel("Year+Quarter")
ax1.set_ylabel("Total LeasedSF (bars)", color="blue")
ax2.set_ylabel("Occupancy Proportion (line)", color="red")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMENTARY:
# This can help show if occupancy is correlated with new large leases.
# If occupancy is dropping, we might see fewer big lease signings or
# we might see big signings but still overall occupancy remains low, etc.

# =========== (C) Rent vs. Lease Size by Sector (If sector or industry columns exist) ============
# Let's assume your leases table has something like 'internal_industry' or 'sector'.
# We'll do a quick boxplot or scatter to see if certain sectors have higher rents.

if "internal_industry" in large_leases.columns:
    # We want to see how 'overall_rent' compares across industries
    # We can do a boxplot for each industry (top 5 industries by frequency).
    top5_industries = (large_leases["internal_industry"]
                       .value_counts()
                       .head(5)
                       .index
                      )
    subset_c = large_leases[large_leases["internal_industry"].isin(top5_industries)].copy()

    # Just to ensure numeric
    subset_c["overall_rent"] = pd.to_numeric(subset_c["overall_rent"], errors="coerce")

    # Boxplot
    plt.figure(figsize=(8,5))
    subset_c.boxplot(column="overall_rent", by="internal_industry", grid=False)
    plt.title("Overall Rent by Industry (Large Leases)")
    plt.suptitle("")  # remove the default boxplot title
    plt.xlabel("Industry")
    plt.ylabel("Overall Rent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMENTARY:
# This reveals which industries (like tech vs. legal vs. finance) might pay higher overall rent.
# Possibly "finance" has higher average rent or "tech" is spread out with outliers, etc.

# =========== (D) Potential Impact of Unemployment on Large-Lease Volume ============
# We can do a correlation between average unemployment_rate and total leasedSF per year

group_d = large_leases.groupby("year").agg({
    "leasedSF": "sum",
    "unemployment_rate": "mean"
}).reset_index()

plt.figure(figsize=(6,4))
plt.scatter(group_d["unemployment_rate"], group_d["leasedSF"], color="green")
plt.title("Large-Lease Volume vs. Unemployment Rate (by Year)")
plt.xlabel("Avg Unemployment Rate")
plt.ylabel("Total LeasedSF (≥10k)")
plt.tight_layout()
plt.show()

# COMMENTARY:
# You might see an inverse relationship (higher unemployment => lower lease signings),
# or it might be inconclusive. A correlation test could be done with e.g.:
corr_value = group_d[["leasedSF","unemployment_rate"]].corr().iloc[0,1]
print(f"\nCorrelation between total leasedSF and unemployment_rate by year: {corr_value:.3f}")

# ---------------------------------------------------------------------
# 5) NEXT STEPS / SUGGESTIONS
# ---------------------------------------------------------------------
"""
- Focus on a single major market or cluster of markets, since data might be sparser outside the top markets.
- Filter to 'internal_industry' in ['tech','legal','financial'] if those exist, and compare how each sector’s
  lease activity changes over time.
- Investigate how direct vs. sublet space changes across quarters (if columns like 'direct_available_space',
  'sublet_available_space' are present).
- Use American Community Survey data (commute times, population changes) to see if there's a relationship
  with the location of new large leases. Possibly at a state or MSA level.
- Consider AI tools or regression to model 'overall_rent' based on occupancy, unemployment, year, etc.
- If submarket data is present, do a deeper dive on submarket-level differences (CBD vs. suburban).
- Create maps or geospatial plots if you have lat/long or region boundaries, to see the distribution
  of large leases.

Overall, the idea is to:
(1) Subset large leases,
(2) Merge occupancy/rent/unemployment for context,
(3) Visualize trends in time,
(4) Compare sectors or submarkets,
(5) Explore deeper correlations or cause-effect hypotheses.
"""



# 1) SUBMARKET BREAKDOWN WITHIN A SINGLE CITY
#    Example: Identify which submarkets are seeing the most large-lease activity.
##############################################################################
if "internal_submarket" in large_leases.columns and "market" in large_leases.columns:
    city_of_interest = "atlanta"  # pick a city from your 'market' column
    city_data = large_leases[ large_leases["market"].str.lower() == city_of_interest ]
    
    # Summarize total leasedSF by submarket
    submarket_sum = (city_data
                     .groupby("internal_submarket")["leasedSF"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(10))  # top 10 submarkets by total SF

    # Plot
    plt.figure(figsize=(8,4))
    submarket_sum.plot(kind="bar", color="teal")
    plt.title(f"Top 10 Submarkets by Total LeasedSF (≥10k) in {city_of_interest.title()}")
    plt.ylabel("Total LeasedSF")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    """
    STORY / INSIGHT:
    - Within a big city, certain submarkets (e.g. Midtown, Buckhead, Perimeter for Atlanta)
      might attract more large leases than others.
    - Clients wanting a submarket with strong leasing momentum might prefer these top ones,
      or they might look for cheaper deals in lower-ranked submarkets.
    """

# 2) CORRELATION: OCCUPANCY VS. OVERALL RENT
#    Are high-occupancy quarters also commanding higher rents?
##############################################################################
if "occupancy_proportion" in large_leases.columns and "overall_rent" in large_leases.columns:
    # We only want rows that have both occupancy_proportion and overall_rent
    corr_data = large_leases.dropna(subset=["occupancy_proportion","overall_rent"]).copy()
    # Convert to numeric if not already
    corr_data["overall_rent"] = pd.to_numeric(corr_data["overall_rent"], errors="coerce")
    corr_data["occupancy_proportion"] = pd.to_numeric(corr_data["occupancy_proportion"], errors="coerce")

    # Group by (year, quarter) or just take direct row-level correlation
    # Here, let's do a row-level scatter:
    plt.figure(figsize=(6,5))
    plt.scatter(corr_data["occupancy_proportion"], corr_data["overall_rent"], alpha=0.3, color="darkorange")
    plt.title("Occupancy Proportion vs. Overall Rent (Large Leases)")
    plt.xlabel("Occupancy Proportion")
    plt.ylabel("Overall Rent")
    plt.tight_layout()
    plt.show()

    # Calculate correlation coefficient
    cor_val = corr_data[["occupancy_proportion","overall_rent"]].corr().iloc[0,1]
    print(f"Correlation(occupancy_proportion, overall_rent) = {cor_val:.3f}")

    """
    STORY / INSIGHT:
    - A positive correlation might suggest that quarters/markets with higher occupancy also have
      higher rents. A weak or negative correlation suggests other factors are at play.
    - This can guide a client to see if strong occupancy means they’ll pay premium rents.
    """

# 3) TOP COMPANIES BY TOTAL SF ACROSS MULTIPLE MARKETS
#    If 'company_name' is present, see which companies are expanding widely.
##############################################################################
if "company_name" in large_leases.columns and "market" in large_leases.columns:
    # Summarize total leasedSF per company across all markets
    comp_sum = (large_leases
                .groupby("company_name")["leasedSF"]
                .sum()
                .sort_values(ascending=False))
    
    # Plot top 10
    top10 = comp_sum.head(10)
    plt.figure(figsize=(8,4))
    top10.plot(kind="bar", color="purple")
    plt.title("Top 10 Companies by Total LeasedSF (≥10k)")
    plt.ylabel("LeasedSF")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    """
    STORY / INSIGHT:
    - Identify major players in the dataset. A single company might be driving big expansions.
    - If you see a well-known brand, you can investigate which markets or submarkets
      they occupy, to advise other clients who want to be near (or far from) that competitor.
    """

# 4) YEAR-BY-YEAR UNEMPLOYMENT COMPARISON AMONG STATES WITH MOST LEASE ACTIVITY
#    If your 'market' and 'state' columns can align states to markets, we can see
#    how unemployment differs among top states with the most large leases.
##############################################################################
if "state" in large_leases.columns and "year" in large_leases.columns:
    # Which states have the most large-lease activity overall?
    state_counts = large_leases["state"].value_counts().head(5).index.tolist()
    sub_st = large_leases[ large_leases["state"].isin(state_counts) ].copy()
    
    # We'll group the UNEMP data by year+state for an average unemployment
    # (Alternatively, we can just re-merge or rely on df_merged if it has unemployment_rate.)
    # If you have it in df_merged, you can do:
    if "unemployment_rate" in sub_st.columns:
        st_unemp = (sub_st.groupby(["year","state"])["unemployment_rate"]
                    .mean()
                    .reset_index())

        # Plot a line for each state over the years
        pivot_unemp = st_unemp.pivot(index="year", columns="state", values="unemployment_rate")
        pivot_unemp.plot(marker="o", figsize=(8,4))
        plt.title("Unemployment Rate by Year in Top 5 States (Large Leases)")
        plt.xlabel("Year")
        plt.ylabel("Unemployment Rate")
        plt.xticks(range(pivot_unemp.index.min(), pivot_unemp.index.max()+1), rotation=45)
        plt.tight_layout()
        plt.show()

        """
        STORY / INSIGHT:
        - Compare how unemployment moves in these top 5 states for large leases.
        - Possibly identify states that recovered faster from economic downturns,
          or that had big layoffs yet still see big leasing. Contrasts can guide
          a client deciding between states.
        """

# 5) SUBLET SPACE vs. DIRECT SPACE (If your minimal-clean data has these columns)
#    Sometimes sublet activity surges after layoffs or WFH expansions.
##############################################################################
possible_cols = set(large_leases.columns.str.lower())
if "direct_available_space" in possible_cols and "sublet_available_space" in possible_cols:
    # Summarize the ratio of sublet vs direct for large leases
    large_leases["sublet_available_space"] = pd.to_numeric(large_leases["sublet_available_space"], errors="coerce")
    large_leases["direct_available_space"] = pd.to_numeric(large_leases["direct_available_space"], errors="coerce")

    group_sublet = (large_leases.groupby(["year","quarter"])
                    [["sublet_available_space","direct_available_space"]]
                    .sum()
                    .reset_index())
    group_sublet["year_quarter"] = group_sublet["year"].astype(str) + "Q" + group_sublet["quarter"].astype(str)

    # Plot side-by-side bars
    plt.figure(figsize=(10,5))
    x = range(len(group_sublet))
    width = 0.35
    plt.bar([xi - width/2 for xi in x],
            group_sublet["direct_available_space"], width=width, color="green", label="Direct Avail Space")
    plt.bar([xi + width/2 for xi in x],
            group_sublet["sublet_available_space"], width=width, color="orange", label="Sublet Avail Space")

    plt.xticks(x, group_sublet["year_quarter"], rotation=45)
    plt.title("Direct vs. Sublet Available Space Over Time (Large Leases)")
    plt.ylabel("SqFt")
    plt.legend()
    plt.tight_layout()
    plt.show()

