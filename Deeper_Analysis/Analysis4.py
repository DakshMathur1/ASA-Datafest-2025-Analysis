import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ---------------------------------------------------------------------
    # 1) LOAD & MERGE MINIMAL-CLEAN DATA
    #    (Same pattern as before, with merges on (market,year,quarter) & (state,year))
    # ---------------------------------------------------------------------
    leases      = pd.read_csv("Leases_minimalclean.csv")
    occupancy   = pd.read_csv("Occupancy_minimalclean.csv")
    price_avail = pd.read_csv("PriceAvail_minimalclean.csv")
    unemp       = pd.read_csv("Unemployment_minimalclean.csv")

    # Merge occupancy
    df_merged = pd.merge(
        leases,
        occupancy[["market","year","quarter","occupancy_proportion"]],
        on=["market","year","quarter"], how="left"
    )
    # Merge price_avail
    df_merged = pd.merge(
        df_merged,
        price_avail[["market","year","quarter","overall_rent"]],
        on=["market","year","quarter"], how="left"
    )
    # Merge unemp
    df_merged = pd.merge(
        df_merged,
        unemp[["state","year","unemployment_rate"]],
        on=["state","year"], how="left"
    )

    # If we have overall_rent_x / overall_rent_y, unify them
    if "overall_rent_x" in df_merged.columns or "overall_rent_y" in df_merged.columns:
        df_merged["overall_rent"] = (
            df_merged.get("overall_rent_x", pd.Series(dtype=float))
            .fillna(df_merged.get("overall_rent_y", pd.Series(dtype=float)))
        )
        df_merged.drop(columns=["overall_rent_x","overall_rent_y"], errors="ignore", inplace=True)

    # Convert numeric columns
    df_merged["leasedSF"] = pd.to_numeric(df_merged["leasedSF"], errors="coerce")
    df_merged["overall_rent"] = pd.to_numeric(df_merged["overall_rent"], errors="coerce")
    df_merged["occupancy_proportion"] = pd.to_numeric(df_merged["occupancy_proportion"], errors="coerce")
    df_merged["unemployment_rate"] = pd.to_numeric(df_merged["unemployment_rate"], errors="coerce")

    print("Final merged shape:", df_merged.shape)
    print("Columns:", df_merged.columns.tolist())

    # ---------------------------------------------------------------------
    # 2) SUBMARKET-LEVEL DEEP DIVE
    #    We'll look at total leasedSF by submarket, YOY changes, etc.
    # ---------------------------------------------------------------------
    if "internal_submarket" in df_merged.columns:
        submkt_data = df_merged.groupby(["internal_submarket","year"], dropna=False)["leasedSF"].sum().reset_index()
        submkt_data.sort_values(["internal_submarket","year"], inplace=True)

        # Create a lag for yoy comparison
        submkt_data["leasedSF_lag"] = submkt_data.groupby("internal_submarket")["leasedSF"].shift(1)
        submkt_data["leasedSF_yoy_pct"] = (
            (submkt_data["leasedSF"] - submkt_data["leasedSF_lag"])
            / submkt_data["leasedSF_lag"] * 100
        )

        # Let’s pick a single submarket to illustrate
        chosen_submkt = "midtown"  # adjust to match your data
        chosen_df = submkt_data[submkt_data["internal_submarket"].str.lower() == chosen_submkt]
        print(f"\n--- Submarket Deep Dive: {chosen_submkt.title()} ---")
        print(chosen_df)

        # Plot yoy % for that submarket
        plt.figure(figsize=(6,4))
        plt.plot(chosen_df["year"], chosen_df["leasedSF_yoy_pct"], marker="o")
        plt.title(f"YoY % Change in LeasedSF: {chosen_submkt.title()}")
        plt.xlabel("Year")
        plt.ylabel("YoY %")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # 3) RELATE UNEMPLOYMENT TO LEASING ACTIVITY
    #    We'll group by (year) or (year,quarter) and see correlation.
    # ---------------------------------------------------------------------
    # Example: total leasedSF per year vs. avg unemployment_rate per year
    group_unemp = df_merged.groupby("year").agg({
        "leasedSF": "sum",
        "unemployment_rate": "mean"
    }).reset_index()

    # Scatter plot
    plt.figure(figsize=(6,4))
    plt.scatter(group_unemp["unemployment_rate"], group_unemp["leasedSF"], color="green")
    plt.title("Total LeasedSF vs. Unemployment Rate (by Year)")
    plt.xlabel("Avg Unemployment Rate")
    plt.ylabel("Total LeasedSF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print correlation
    corr_val = group_unemp[["leasedSF","unemployment_rate"]].corr().iloc[0,1]
    print(f"\nCorrelation between total leasedSF and unemployment_rate: {corr_val:.3f}")

    # ---------------------------------------------------------------------
    # 4) SECTOR / INDUSTRY ANALYSIS
    #    If there's an 'internal_industry' or 'sector' column
    # ---------------------------------------------------------------------
    if "internal_industry" in df_merged.columns:
        # Summarize total leasedSF by industry
        ind_data = df_merged.groupby("internal_industry")["leasedSF"].sum().sort_values(ascending=False)
        print("\n--- Top 10 Industries by LeasedSF ---")
        print(ind_data.head(10))

        # Quick bar chart
        top10_ind = ind_data.head(10)
        plt.figure(figsize=(8,4))
        top10_ind.plot(kind="bar", color="purple")
        plt.title("Top 10 Industries by Total LeasedSF")
        plt.ylabel("LeasedSF")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # 5) QUICK REGRESSION
    #    We'll do a simple OLS: overall_rent ~ occupancy_proportion + unemployment_rate + ...
    #    Using statsmodels. You must have statsmodels installed.
    # ---------------------------------------------------------------------
    #  - Filter rows that have no NaN for these columns
    #  - We'll also add a constant for the intercept
    import statsmodels.api as sm

    reg_cols = ["overall_rent","occupancy_proportion","unemployment_rate","leasedSF"]
    reg_df = df_merged.dropna(subset=reg_cols).copy()

    # We'll predict log(overall_rent) as an example:
    reg_df["ln_rent"] = np.log(reg_df["overall_rent"])

    # Explanatory variables: occupancy_proportion, unemployment_rate, and maybe log(leasedSF)
    reg_df["ln_leasedSF"] = np.log(reg_df["leasedSF"] + 1)  # +1 in case of zero
    X = reg_df[["occupancy_proportion","unemployment_rate","ln_leasedSF"]]
    X = sm.add_constant(X)  # add intercept
    y = reg_df["ln_rent"]

    model = sm.OLS(y, X).fit()
    print("\n--- Quick Regression Results: ln(overall_rent) ~ occupancy + unemp + ln_leasedSF ---")
    print(model.summary())

    """
    INTERPRETATION:
    - Coeff on occupancy_proportion: how a 1-unit change in occupancy (100%) affects ln(rent).
    - Coeff on unemployment_rate: effect on ln(rent).
    - Coeff on ln_leasedSF: effect of the scale of a lease on rent levels.
    - R-squared tells how much variance is explained.

    This is purely illustrative. Adjust columns or add dummy variables for region, submarket, etc.
    """

if __name__ == "__main__":
    main()
