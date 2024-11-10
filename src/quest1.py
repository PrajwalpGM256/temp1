import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diamonds dataset
diamonds_url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
diamonds = pd.read_csv(diamonds_url)

# Plotting price vs. carat
plt.figure(figsize=(10, 6))
sns.scatterplot(x='carat', y='price', data=diamonds)
plt.title('Price vs. Carat')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.savefig('figs/Q1_Price vs Carat')
plt.show()

