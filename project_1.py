import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

path = r"C:\Desktop\CUSTOMER CHURN\WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(path)

print("=== TELCO CUSTOMER CHURN DATASET EXPLORATION ===\n")

# basic dataset info
print("1. DATASET SHAPE AND BASIC INFO: ")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.to_list()}")
print("\n" + "="*50 + "\n")



# display first 5 rows
print("2. FIRST 5 ROWS: ")
print(df.head(5))
print("\n" + "="*50 + "\n")

# data types and missing values
print("3. DATA TYPES AND MISSING VALUES: ")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum()) # no missing values
print("\n" + "="*50 + "\n")

# target value distribution
print("4. TARGET VARIABLE DISTRIBUTION: ")
print("Churn Distribution:")
print(df['Churn'].value_counts())
print("Churn percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)
print("\n" + "="*50 + "\n")



# numerical vs categorical features
print("5. FEATURE TYPES: ")
numerical_features = df.select_dtypes(include=[np.number]).columns.to_list()
categorical_features = df.select_dtypes(include=["object"]).columns.to_list()
print(f"Numerical Features ({len(numerical_features)}) : {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}) : {categorical_features}")
print("\n" + "="*50 + "\n")



# basic statistics for numerical features
print("6. NUMERICAL FEATURE STATISTICS:")
print(df[numerical_features].describe())
print("\n" + "="*50 + "\n")

# check for any unusual values
print("7. CHECK FOR UNUSUAL VALUES:")
for col in df.columns:
    unique_vals = df[col].nunique()
    # nunique ---> returns the number of unique values in that column
    print(f"{col}:{unique_vals} unique values")
    if unique_vals < 10:
        print(f" Values : {df[col].unique()}")
        # df[col].unique() ---> Returns a NumPy array of all unique values in that column
    print()

print("=== DATA PREPROCESSING ===\n")

# handling the 'TotalCharges' columns (common issue in this dataset ^^')
print("1. FIXING TOTAL CHARGES COLUMN:")
print(f"TotalCharges dtype before: {df['TotalCharges'].dtype}")
print(f"Sample TotalCharges values: {df['TotalCharges'].head()}")

# converting TotalCharges column to numeric since it's often stores as string
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
# errors='coerce' ---> If any value cannot be converted to a number (e.g., empty string, text like "abc"), it is replaced with NaN instead of throwing an error.
print(f"TotalCharges dtype after: {df['TotalCharges'].dtype}")

# checking for data loss that might have occurred during conversion
print(f"Missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")

# Missing values in TotalCharges after conversion to numeric form: 11

# handling missing values by replacing them with median
if df["TotalCharges"].isnull().sum() > 0:
    print("Replacing missing TotalCharges with median...")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print(f"{df['TotalCharges'].isnull().sum()} missing values in TotalCharges")
print("\n" + "="*50 + "\n")


# separating features and target
print("2. SEPARATING FEATURES AND TARGET")
# removing customer id as it's not useful to us for prediction
x = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

print(f"Features Shape: {x.shape}")
print(f"Target Shape: {y.shape}")
print(f"Target: {y.name}")
print("\n" + "="*50 + "\n")

# encoding categorical variables
print("3. ENCODING CATEGORICAL VARIABLES:")

# making a copy to work with
x_processed = x.copy()

# identify categorical columns
categorical_cols = x_processed.select_dtypes(include=['object']).columns.to_list()
numerical_cols = x_processed.select_dtypes(include=[np.number]).columns.to_list()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# label encode categorical variables
# One-hot encode categorical features
x_processed = pd.get_dummies(x_processed, columns=categorical_cols)

# encode target variable
y_encoded = LabelEncoder().fit_transform(y)
print(f"Target encoded : No = 0, Yes = 1")
print("\n" + "="*50 + "\n")

# scaling numerical features
print("4. SCALING NUMERICAL FEATURES:")
sc = StandardScaler()
x_processed[numerical_cols] = sc.fit_transform(x_processed[numerical_cols])

print("Numerical Features scaled to mean = 0, std = 1")
print(f"Sample Scaled Values: {x_processed[numerical_cols].head(5)}")
print("\n" + "="*50 + "\n")

print(f"Final shape of processed features: {x_processed.shape}")
print(f"Encoded target shape: {y_encoded.shape}")

import joblib
# save the scaler
joblib.dump(sc, 'scaler.pkl')

# save the label encoder for the target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
joblib.dump(le_y, 'label_encoder_y.pkl')

# save the column order (useful for inference)
joblib.dump(x_processed.columns.tolist(), 'feature_columns.pkl')


# split the data
print("5. SPLITTING THE DATA INTO TRAIN/TEST:")
x_train, x_test, y_train, y_test = train_test_split(
    x_processed, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded # ensures that the split is balanced 
    # stratify ---> Splits the data such that the same proportion of classes (e.g., churned vs not churned) appears in both training and test sets.
    # Use it when:
    # - You're doing classification
    # - Your target class distribution is not perfectly balanced
)



print(f"Train set: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Train target distribution: {np.bincount(y_train)}")
print(f"Test target distribution: {np.bincount(y_test)}")
# bincount ---> Counts the number of occurrences of each integer value in the array.
# Use this whenever you're:
# - Diagnosing class imbalance
# - Verifying stratified splits
# - Sanity-checking model input/output shapes
print("\n" + "="*50 + "\n")

# final check
print("6. FINAL PREPROCESSED DATA SUMMARY: ")
print(f"TOTAL FEATURES : {x_processed.shape[1]}")
print(f"All features are now numerical : {x_processed.dtypes.unique()}")
print(f"No missing values : {x_processed.isnull().sum().sum() == 0}")
print(f"Target is binary encoded : {np.unique(y_encoded)}")

print("\n Data Preprocessing Complete! Ready for algorithm implementation!")

# Algorithm implementation and comparison
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

print("=== ALGORITHM IMPLEMENTATION AND COMPARISON ===\n")

# storing our models in a dictionary
models = {
    'Logistic Regression' : LogisticRegression(random_state=42),
    #'Multiple Logistic Regression': LogisticRegression(multi_class='multinomial', random_state=42, max_iter=175),
    #'OVR Logistic Regression': LogisticRegression(multi_class='ovr', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, kernel='linear'),
    'Naive Bayes': GaussianNB(),
    'K Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost' : XGBClassifier(eval_metric = 'logloss', random_state = 42)
}

# dictionary to store results
results = {}
print("Training and Evaluating Models...\n")

# training and evaluating each model
for name, model in models.items():
    print(f"Training...{name}")

    # recording training time
    start_time = time.time()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # calculate training time
    training_time = time.time() - start_time

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall' : recall,
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred
    }

    print(f"    {name} completed training in {training_time:.2f} seconds")
    print(f"    Accuracy : {accuracy:.3f}")
    print(f"    F1-score: {f1:.3f}")
    print()

print("=" * 60)
print("COMPARATIVE ANALYSIS RESULTS")
print("=" * 60)

# creating dataframe for easy comparison
results_df = pd.DataFrame({
    'Algorithm': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()],
    'Training Time (s)': [results[model]['training_time'] for model in results.keys()] 
})

# sorting by f1-score (good for imbalanced datasets)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\n PERFORMANCE RANKING (by F1-Score):")
print(results_df.round(3))

print("\n BEST PERFORMING MODEL:")
best_model = results_df.iloc[0]['Algorithm']
print(f"Winner: {best_model}")
print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.3f}")
print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.3f}")

# create visualizations
fig, axes = plt.subplots(2, 2, figsize = (15, 10))

# 1. Accuracy Comparison
axes[0, 0].bar(results_df['Algorithm'], results_df['Accuracy'])
axes[0, 0].set_title("Accuracy Comparison")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].tick_params(axis = 'x', rotation = 45)

# 2. F1-Score Comparison
axes[0, 1].bar(results_df['Algorithm'], results_df['F1-Score'])
axes[0, 1].set_title("F1-Score Comparison")
axes[0, 1].set_ylabel("F1-Score")
axes[0, 1].tick_params(axis = 'x', rotation = 45)

# 3. Training Time Comparison
axes[1, 0].bar(results_df['Algorithm'], results_df['Training Time (s)'])
axes[1, 0].set_title("Training Time Comparison")
axes[1, 0].set_ylabel("Time (seconds)")
axes[1, 0].tick_params(axis = 'x', rotation = 45)

# 4. Precision vs Recall
axes[1, 1].scatter(results_df['Recall'], results_df['Precision'], s=100)
for i, txt in enumerate(results_df['Algorithm']):
    axes[1, 1].annotate(txt, (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)


axes[1, 1].set_title('Precision vs Recall')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')

plt.tight_layout()
plt.show()

print("\n DETAILED ANALYSIS")
print("=" * 40)

# detailed analysis for the best model
best_model_name = results_df.iloc[0]['Algorithm']
best_model_predictions = results[best_model_name]['predictions']

print(f"\n DETAILED ANALYSIS OF {best_model_name.upper()}:")
print(classification_report(y_test, best_model_predictions, target_names=['No Churn', 'Churn']))


# Confusion Matrix for the best model
cm = confusion_matrix(y_test, best_model_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


print("\n KEY INSIGHTS:")
print("=" * 40)
print(f"• Best performing algorithm: {best_model_name}")
print(f"• Dataset is imbalanced (73% No Churn, 27% Churn)")
print(f"• F1-Score is important for imbalanced datasets")
print(f"• Consider the trade-off between precision and recall")
print(f"• Training time varies significantly between algorithms")
