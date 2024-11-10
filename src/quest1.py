# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data file (replace 'your_data_file.csv' with the actual file name)
df = pd.read_csv('data/diamonds.csv')

# Display the first few rows to verify the data
print(df.head())

# Create a scatter plot of price vs. carat
plt.figure(figsize=(10, 6))
sns.scatterplot(x='carat', y='price', data=df, alpha=0.6)
plt.title('Scatter Plot of Price vs. Carat')
plt.xlabel('Carat')
plt.ylabel('Price')

plt.savefig('figs/Q1_Price vs Carat')
plt.show()

