
<div align="center">
      <H1> Crimes-Against-Women-in-India-2001-2021-</H1>
<H2>
</H2>  
     </div>

<body>
<p align="center">
  <a href="mailto:arifmiahcse952@gmail.com"><img src="https://img.shields.io/badge/Email-arifmiah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/Arif-miad"><img src="https://img.shields.io/badge/GitHub-%40ArifMiah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/arif-miah-8751bb217/"><img src="https://img.shields.io/badge/LinkedIn-Arif%20Miah-blue?style=flat-square&logo=linkedin"></a>

 
  
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801998246254-green?style=flat-square&logo=whatsapp">
  
</p>
About Dataset
Crimes against women in India from 2001 to 2021
This data is collated from https://data.gov.in. It has state-wise data on the various crimes committed against women between 2001 to 2021. Some crimes that are included are Rape, Kidnapping and Abduction, Dowry Deaths etc.

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import plotly.express as px

# Ignore all warnings
warnings.filterwarnings('ignore')
crimes_df = pd.read_csv('/kaggle/input/crimes-against-women-in-india-2001-2021/CrimesOnWomenData.csv')
description_df = pd.read_csv('/kaggle/input/crimes-against-women-in-india-2001-2021/description.csv')

# Display the first few rows of each dataset
print("CrimesOnWomenData.csv - First 5 Rows:")
print(crimes_df.head())

print("\nDescription.csv - First 5 Rows:")
print(description_df.head())
# Pivot the data for heatmap
heatmap_data = crimes_df_cleaned.pivot_table(values='Rape Cases', index='State', columns='Year', aggfunc='sum', fill_value=0)

plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_data, cmap="YlGnBu", linecolor='white', linewidths=0.5)
plt.title('Heatmap of Rape Cases by State and Year')
plt.xlabel('Year')
plt.ylabel('State')
plt.show()
```
![image](https://github.com/Arif-miad/Crimes-Against-Women-in-India-2001-2021-/blob/main/crime.png)

```python
# Group by year and sum up all crime types
crime_trend = crimes_df_cleaned.groupby('Year').sum()

# Plotting the trend of different crimes over the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=crime_trend)
plt.title('Trend of Crimes Against Women (2001-2021)')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```
![image](https://github.com/Arif-miad/Crimes-Against-Women-in-India-2001-2021-/blob/main/crime2.png)
```python
plt.figure(figsize=(12, 8))
correlation_matrix = crimes_df_cleaned.drop(['State'], axis=1).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Different Crimes')
plt.show()
```

![image](https://github.com/Arif-miad/Crimes-Against-Women-in-India-2001-2021-/blob/main/crime3.png)
```python
import seaborn as sns

# Select top N states by total crime numbers
top_n_states = crimes_df_cleaned.groupby('State').sum().nlargest(6, 'Rape Cases').index

# Filter data for these states
filtered_df = crimes_df_cleaned[crimes_df_cleaned['State'].isin(top_n_states)]

# Create a facet grid to show trends over the years
g = sns.FacetGrid(filtered_df, col="State", col_wrap=3, height=4)
g.map(sns.lineplot, 'Year', 'Rape Cases')
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Number of Rape Cases")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Rape Cases Trends in Top 6 States', fontsize=16)
plt.show()
```

![image](https://github.com/Arif-miad/Crimes-Against-Women-in-India-2001-2021-/blob/main/__results___16_0.png)




