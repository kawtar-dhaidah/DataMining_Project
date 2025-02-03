# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

file_path = "cleaned_healthcare_dataset6.csv"
df = pd.read_csv(file_path)

# Step 1: Describe the dataset
print("Dataset Overview:")
print(df.info())  # Details on columns, data types, and non-null values
print("\nStatistical Summary of Numeric Columns:")
print(df.describe())  # Summary statistics for numeric columns

# Step 2: Check and clean the 'Age' column
# Check for missing values
missing_ages = df['Age'].isnull().sum()
print(f"Missing values in 'Age' column: {missing_ages}")

# Ensure 'Age' column is numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Fill missing values with the mean of the 'Age' column
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Step 3: Visualize the data (Box Plot)
plt.style.use('ggplot')  # Apply ggplot style
fig, ax = plt.subplots()

# Define numeric columns for visualization
numeric_columns = ['Age', 'Billing Amount', 'Room Number']
df_clean = df[numeric_columns].dropna()

# Create a box plot
ax.boxplot([df_clean[col] for col in numeric_columns if col in df.columns],
           vert=False, showmeans=True, meanline=True,
           labels=[col for col in numeric_columns if col in df.columns],
           patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})

plt.title("Box Plot of Numeric Variables")
plt.show()

# Step 4: Draw histograms and fit normal distributions for numeric variables
for col in numeric_columns:
    if col in df.columns:
        # Fit a normal distribution
        mu, std = norm.fit(df[col].dropna())  # Avoid NaN values for fitting

        # Plot histogram
        plt.hist(df[col].dropna(), bins=10, density=True, alpha=0.6, color='blue')

        # Plot normal distribution curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)

        plt.title(f"{col} - Fit Values: μ={mu:.2f}, σ={std:.2f}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.show()

# Step 5: Correlation Analysis
# Sort by date columns and drop irrelevant ones
date_columns = ['Date of Admission', 'Discharge Date']
if all(col in df.columns for col in date_columns):
    df_sorted = df.sort_values(date_columns[0])  # Sort by 'Date of Admission'
    df_sorted.drop(columns=date_columns, inplace=True, errors='ignore')  # Remove date columns for correlation
else:
    df_sorted = df.copy()



# Heatmap of correlations
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_sorted.select_dtypes(include=[np.number]).corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()


df_numeric = df[numeric_columns]
df_numeric = df_numeric.dropna()

# Perform covariance and Pearson correlation analysis between each pair of numeric columns
for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns)):
        # Covariance calculation
        cov_xy = df_numeric[numeric_columns[i]].cov(df_numeric[numeric_columns[j]])
        print(f"Covariance of {numeric_columns[i]} and {numeric_columns[j]}: {cov_xy}")
        
        # Pearson correlation calculation
        try:
            r, p = scipy.stats.pearsonr(df_numeric[numeric_columns[i]], df_numeric[numeric_columns[j]])
            print(f"Correlation of {numeric_columns[i]} and {numeric_columns[j]}: r = {r:.2f}, p-value = {p:.5f}")
        except ValueError:
            print(f"Correlation calculation failed for {numeric_columns[i]} and {numeric_columns[j]} due to non-finite values.")



