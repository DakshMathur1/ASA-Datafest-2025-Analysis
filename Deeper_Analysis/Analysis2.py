import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1) LOAD THE ORIGINAL LEASES
# -------------------------
leases = pd.read_csv('Leases.csv')  # <-- The original, uncleaned Leases file
print("Leases.csv loaded. Shape =", leases.shape)

# Inspect columns
print("Columns in original Leases.csv:", leases.columns.tolist())

# -------------------------
# 2) COUNT LARGE LEASES (≥ 10,000 SF)
# -------------------------
# Assuming the column is named 'leasedSF'. 
# If it's named differently (e.g. 'leased_sqft'), change below:
count_all = len(leases)
count_large = (leases['leasedSF'] >= 10000).sum()
print(f"Total rows in original Leases.csv: {count_all}")
print(f"Number of large leases (≥ 10k SF): {count_large}")

# Create a subset of just large leases
large_leases = leases[leases['leasedSF'] >= 10000].copy()
print("Shape of large_leases subset:", large_leases.shape)

# -------------------------
# 3) BASIC ANALYSIS: 
#    Summarize large leases by (year, quarter)
# -------------------------
# If your original file has columns 'year' and 'quarter'
# (If they're spelled differently, adjust the code.)
pivot_lg = large_leases.groupby(['year','quarter'])['leasedSF'].sum().reset_index()

# Print top rows
print("\nPivot of large leases by year+quarter (sum of leasedSF):")
print(pivot_lg.head(15))

# Optional: Plot as a bar chart
pivot_lg['year_quarter'] = pivot_lg['year'].astype(str) + "Q" + pivot_lg['quarter'].astype(str)
pivot_lg.plot(x='year_quarter', y='leasedSF', kind='bar',
              title='Total LeasedSF (≥10k) by Year+Quarter',
              figsize=(10,5))
plt.xticks(rotation=45)
plt.ylabel('LeasedSF')
plt.show()



