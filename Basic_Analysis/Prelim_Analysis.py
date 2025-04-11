import pandas as pd
import matplotlib.pyplot as plt

leases = pd.read_csv('cleaned_leases_full_dynamic.csv')
occupancy = pd.read_csv('cleaned_occupancy_full_dynamic.csv')
price_avail = pd.read_csv('cleaned_price_avail_full_dynamic.csv')
unemp = pd.read_csv('cleaned_unemployment_full_dynamic.csv')

print("LEASES COLUMNS:", leases.columns.tolist())
print("OCCUPANCY COLUMNS:", occupancy.columns.tolist())
print("PRICE_AVAIL COLUMNS:", price_avail.columns.tolist())
print("UNEMP COLUMNS:", unemp.columns.tolist())

# 1) LEASES: sum leased_sqft if it exists
if all(col in leases.columns for col in ['leased_sqft','market','year_quarter']):
    pivot_leases = leases.groupby(['year_quarter','market'])['leased_sqft'].sum().unstack()
    pivot_leases.plot(title='Leased SqFt by Market + Quarter', figsize=(8,4))
    plt.xticks(rotation=45)
    plt.ylabel('SqFt')
    plt.show()

# 2) OCCUPANCY: average occupancy_rate if it exists
if all(col in occupancy.columns for col in ['occupancy_rate','market','year_quarter']):
    pivot_occ = occupancy.groupby(['year_quarter','market'])['occupancy_rate'].mean().unstack()
    pivot_occ.plot(title='Occupancy Rate by Market + Quarter', figsize=(8,4))
    plt.xticks(rotation=45)
    plt.ylabel('Occupancy Rate')
    plt.show()

# 3) PRICE: if "asking_rent" or "avg_rent" is present
if 'asking_rent' in price_avail.columns:
    pivot_price = price_avail.groupby(['year_quarter','market'])['asking_rent'].mean().unstack()
    pivot_price.plot(title='Asking Rent by Market + Quarter', figsize=(8,4))
    plt.xticks(rotation=45)
    plt.ylabel('Asking Rent')
    plt.show()
elif 'avg_rent' in price_avail.columns:
    pivot_price = price_avail.groupby(['year_quarter','market'])['avg_rent'].mean().unstack()
    pivot_price.plot(title='Average Rent by Market + Quarter', figsize=(8,4))
    plt.xticks(rotation=45)
    plt.ylabel('Average Rent')
    plt.show()

# 4) UNEMP: if "unemployment_rate" is present
if all(col in unemp.columns for col in ['unemployment_rate','year','state']):
    pivot_unemp = unemp.groupby(['year','state'])['unemployment_rate'].mean().unstack()
    pivot_unemp.plot(title='Unemployment Rate by State + Year', figsize=(8,4))
    plt.xticks(rotation=45)
    plt.ylabel('Unemployment Rate')
    plt.show()
