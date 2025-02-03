import pandas as pd
import scipy.stats as stats

# Load the dataset
file_path = 'cleaned_healthcare_dataset6.csv' 
df = pd.read_csv(file_path, delimiter=',')  

# Display dataset structure for debugging
print("Dataset Columns:\n", df.columns)
print("\nSample Data:\n", df.head())

#  One-sample t-test (Testing if the mean of 'Age' is equal to 39)
hypothesized_mean = 39  # Hypothetical mean value for age

if 'Age' in df.columns:
    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(df['Age'].dropna(), popmean=hypothesized_mean)

    print(" One-sample t-test for Age")
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Reject H0: The mean Age is significantly different from 39.")
    else:
        print("Fail to reject H0: No significant difference from 39.")
else:
    print("Column 'Age' not found in the dataset.")

#  Independent two-sample t-test (Testing if the mean 'Billing Amount' differs by 'Admission Type')
if 'Admission Type' in df.columns and 'Billing Amount' in df.columns:
    # Clean the data to remove rows with missing values in 'Billing Amount' or 'Admission Type'
    df_cleaned = df.dropna(subset=['Billing Amount', 'Admission Type']).copy()
    
    # Combine 'Elective' and 'Urgent' categories into 'Non-Emergency'
    df_cleaned['Admission Type'] = df_cleaned['Admission Type'].replace({'Elective': 'Non-Emergency', 'Urgent': 'Non-Emergency'})
    
    # Check the distribution of 'Admission Type' after replacement
    #print(df_cleaned['Admission Type'].value_counts())

    # Ensure there are at least two groups for 'Admission Type'
    if df_cleaned['Admission Type'].nunique() > 1:
        # Create two groups based on 'Admission Type' (e.g., 'Emergency' vs 'Non-Emergency')
        emergency_admission = df_cleaned[df_cleaned['Admission Type'] == 'Emergency']['Billing Amount']
        non_emergency_admission = df_cleaned[df_cleaned['Admission Type'] == 'Non-Emergency']['Billing Amount']

        # Check if both groups have enough data
        if len(emergency_admission) > 0 and len(non_emergency_admission) > 0:
            # Perform independent two-sample t-test
            t_stat, p_value = stats.ttest_ind(emergency_admission, non_emergency_admission, equal_var=False)

            print("\nIndependent two-sample t-test for Billing Amount and Admission Type")
            print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("Reject H0: There is a significant difference in Billing Amount between Emergency and Non-Emergency admissions.")
            else:
                print("Fail to reject H0: No significant difference in Billing Amount between Emergency and Non-Emergency admissions.")
        else:
            print("One of the groups has insufficient data for the t-test.")
    else:
        print("There are not enough categories in 'Admission Type' to perform the t-test.")
else:
    print("Columns 'Admission Type' or 'Billing Amount' not found in the dataset.")
