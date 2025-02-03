import pandas as pd

#Reads the dataset and specifies ; as the delimiter
file_path = "healthcare_dataset1.csv" 
df = pd.read_csv(file_path, delimiter=';')

# Check if dataset is loaded correctly by displaying the first few rows
print(df.head()) 

# Step 1: Check and display duplicates based on all columns
print("\nChecking for duplicate rows based on all columns:")
# Keep=False to show all duplicates
all_duplicate_rows = df[df.duplicated(keep=False)]  
if not all_duplicate_rows.empty:
    print("Duplicate rows:\n", all_duplicate_rows)
    print(f"Total duplicate rows based on all columns: {len(all_duplicate_rows)}")
else:
    print("No duplicate rows found based on all columns.")

# Step 2: Remove duplicates from the entire dataset while keeping the first occurence
df = df.drop_duplicates(keep='first').reset_index(drop=True)
print(f"\nTotal rows after removing duplicates across all columns: {len(df)}")

# Function to remove outliers using IQR
def remove_outliers_iqr(df):
    # Automatically identify numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    
    # Identify outliers
    outliers = ((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Display indices of the rows being removed
    outlier_indices = df[outliers].index.tolist()
    if outlier_indices:
        print(f"Rows removed due to outliers: {outlier_indices}")
    else:
        print("No rows removed due to outliers.")
    
    # Remove outliers
    df_no_outliers = df[~outliers]
    return df_no_outliers

# Apply the IQR outlier removal
df = remove_outliers_iqr(df)
print(f"Total rows after removing outliers using IQR: {len(df)}")



# Step 3: Handle missing values
print("\nHandling missing values:")
# Display missing value summary
missing_values_summary = df.isnull().sum()
print("Missing values summary before cleaning:\n", missing_values_summary)

#  Handle missing values by replacing them with the string "NaN"
print("\nHandling missing values:")
# List of columns to process
date_columns = ['Date of Admission', 'Discharge Date']
categorical_columns = ['Name','Gender','Blood Type','Medical Condition','Doctor','Hospital', 'Insurance Provider', 'Admission Type','Medication','Test Results']
numerical_columns = ['Billing Amount', 'Room Number','Age']

# Replacing missing values in categorical columns with "NaN"
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('NaN')
        print(f"Replaced missing values in categorical column '{col}' with 'NaN'.")

# Replacing missing values in numerical columns with NaN
for col in numerical_columns:
    if col in df.columns:
        df[col] = df[col].fillna("NaN")
        print(f"Replaced missing values in numerical column '{col}' with NaN.")

# Convert date columns to datetime, and delete invalid dates 
for col in date_columns:
    if col in df.columns:
        # Convert to datetime
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove rows where the date is empty
        df = df[df[col].notna()]
        # Format to show only the date part without time
        df[col] = df[col].dt.strftime('%Y-%m-%d')
        print(f"Processed date column '{col}' and removed rows with invalid or empty dates.")

# missing value after cleaning
missing_values_summary_after = df.isnull().sum()
print("Missing values summary after cleaning:\n", missing_values_summary_after)




# Save cleaned data to a new file
cleaned_file_path = "cleaned_healthcare_dataset6.csv"
df.to_csv(cleaned_file_path, index=False)

# Output file location
print(f"Cleaned data saved to: {cleaned_file_path}")
