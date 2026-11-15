# Python code sample: Income and COVID-19 Mortality Analysis
# Data ingestion, cleaning, statistical analysis, and visualization

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

# Ingest data from CSV
covid_raw = pd.read_csv('covid_us_states.csv')
income_raw = pd.read_csv('acs_2019_median_household_income.csv')
population_raw = pd.read_csv('acs_2019_state_population.csv')

# Manage data types and clean COVID-19 records
covid_raw["date"] = pd.to_datetime(covid_raw["date"])
covid_clean = (
    covid_raw
    .drop_duplicates()
    .sort_values(['state', 'date'])
    .groupby('state')
    .tail(1)
    [['state', 'date', 'cases', 'deaths']]
)

# Convert Census income strings to numeric values
income_raw['Label_clean'] = (
    income_raw['Label (Grouping)']
    .astype(str)
    .str.replace('\u00a0', ' ', regex=False)
    .str.strip()
)

median_row = income_raw.loc[
    income_raw['Label_clean'].eq('Median income (dollars)')
].iloc[0]

income = pd.DataFrame([
    {
        "state": column.split('!!')[0],
        "median_household_income_2019": pd.to_numeric(
            str(value).replace(',', '').replace('$', '').strip(),
            errors='coerce'
        )
    }
    for column, value in median_row.items()
    if "!!Households!!Estimate" in column
])

# Convert Census population strings to numeric values
population_row = population_raw.iloc[0]

population = pd.DataFrame([
    {
        "state": column.split('!!')[0],
        "population_2019": pd.to_numeric(
            str(value).replace(',', '').strip(),
            errors='coerce'
        )
    }
    for column, value in population_row.items()
    if column.endswith("!!Estimate")
])

# Merge and wrangle the three datasets
merged = (
    covid_clean
    .merge(income, on='state', how='inner', validate="one_to_one")
    .merge(population, on='state', how='inner', validate="one_to_one")
    # Restrict the analysis to the 50 states and Washington, D.C.
    .query("state != 'Puerto Rico'")
    .copy()
)

merged['cases_per_100k'] = merged['cases'] / merged['population_2019'] * 100000
merged['deaths_per_100k'] = merged['deaths'] / merged['population_2019'] * 100000

# Define a statistical analysis function
def analyze_income_mortality(data):
    income_values = data["median_household_income_2019"]
    mortality_values = data["deaths_per_100k"]

    pearson = pearsonr(income_values, mortality_values)
    spearman = spearmanr(income_values, mortality_values)
    regression = linregress(income_values, mortality_values)

    return {
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "slope": regression.slope,
        "intercept": regression.intercept,
        "r_squared": regression.rvalue**2
    }

# Use the function and output the results
results = analyze_income_mortality(merged)

print(f"Observations: {len(merged)}")
print(
    f"Pearson r: {results['pearson_r']:.4f} "
    f"(p = {results['pearson_p']:.4e})"
)
print(
    f"Spearman rho: {results['spearman_rho']:.4f} "
    f"(p = {results['spearman_p']:.4e})"
)
print(f"Regression slope: {results['slope']:.4f}")
print(f"R-squared: {results['r_squared']:.4f}")

# Visualize the fitted relationship
x = merged["median_household_income_2019"]
y = merged["deaths_per_100k"]
fitted_x = np.linspace(x.min(), x.max(), 100)
fitted_y = results["intercept"] + results["slope"] * fitted_x
plt.figure(figsize=(6.5, 4.5))
plt.scatter(x, y, s=45, alpha=0.75)
plt.plot(fitted_x, fitted_y, color="darkred", linewidth=2)
plt.xlabel("2019 median household income (USD)")
plt.ylabel("COVID-19 deaths per 100,000 residents")
plt.title(
    f"Income and COVID-19 Mortality "
    f"(Pearson r = {results['pearson_r']:.2f})"
)
plt.tight_layout()
plt.show()