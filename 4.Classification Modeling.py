import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'cleaned_healthcare_dataset6.csv'
df = pd.read_csv(file_path, delimiter=',')

# Check if 'Test Results' column exists
if 'Test Results' not in df.columns:
    raise ValueError("Column 'Test Results' not found in the dataset.")

# Define the target variable (binary classification based on 'Test Results')
df['Test Results'] = df['Test Results'].map({'Normal': 0, 'Abnormal': 1}).fillna(0)

# Define features (X) and target (y)
irrelevant_columns = ['Name', 'Test Results', 'Date of Admission', 'Doctor', 'Hospital', 'Room Number', 'Discharge Date']
existing_irrelevant_columns = [col for col in irrelevant_columns if col in df.columns]
X = df.drop(columns=existing_irrelevant_columns)  # Drop only existing columns
y = df['Test Results']

# Handle missing values
numeric_columns = X.select_dtypes(include=['number']).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# Convert categorical features to numeric using Label Encoding
for col in categorical_columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Standardize numeric data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Model 4: K-Nearest Neighbors (KNN)
print("\nTraining K-Nearest Neighbors Model:")
knn_model = KNeighborsClassifier(n_neighbors=10)  
knn_model.fit(X_train, y_train)

# Evaluate K-Nearest Neighbors Model
y_pred_knn = knn_model.predict(X_test)
print("\nK-Nearest Neighbors Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

#Model1: Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))



# Final Comparison
print("\nComparison of Models:")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"K-Nearest Neighbors Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")


# Forecast or Prediction Step
new_data = pd.DataFrame({
    'Age': [33, 70],
    'Blood Type': ['A+', 'A-'],
    'Medical Condition': ['Diabetes', 'Asthma'],
    'Billing Amount': [217, 500],
    'Admission Type': ['Emergency', 'Urgent'],
    'Medication': ['Paracetamol', 'Lipitor']
})

# Preprocess new data
new_data_encoded = pd.get_dummies(new_data, drop_first=True)
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
new_data_scaled = scaler.transform(new_data_encoded)

# Predictions
rf_predictions = rf_model.predict(new_data_scaled)
knn_predictions = knn_model.predict(new_data_encoded)

# Decode predictions 
test_results_mapping = {0: 'Normal', 1: 'Abnormal'}
rf_predictions_decoded = [test_results_mapping[pred] for pred in rf_predictions]
knn_predictions_decoded = [test_results_mapping[pred] for pred in knn_predictions]


print("\nPredictions for New Data:")
print(f"Random Forest Predictions: {rf_predictions_decoded}")
print("K-Nearest Neighbors Predictions for New Data:", knn_predictions_decoded)

