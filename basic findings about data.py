# basic dataset info
'''
Shape: (7043, 21)
Columns: ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
'''
# no missing values
# target value distribution
'''
Churn
No     5174
Yes    1869
Name: count, dtype: int64
Churn percentage:
Churn
No     73.463013
Yes    26.536987
Name: proportion, dtype: float64
'''

# numerical vs categorical features
'''
5. FEATURE TYPES:
Numerical Features (3) : ['SeniorCitizen', 'tenure', 'MonthlyCharges']
Categorical Features (18) : ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']
'''

# check for any unusual values
'''
7. CHECK FOR UNUSUAL VALUES:
customerID:7043 unique values

gender:2 unique values
 Values : ['Female' 'Male']

SeniorCitizen:2 unique values
 Values : [0 1]

Partner:2 unique values
 Values : ['Yes' 'No']

Dependents:2 unique values
 Values : ['No' 'Yes']

tenure:73 unique values

PhoneService:2 unique values
 Values : ['No' 'Yes']

MultipleLines:3 unique values
 Values : ['No phone service' 'No' 'Yes']

InternetService:3 unique values
 Values : ['DSL' 'Fiber optic' 'No']

OnlineSecurity:3 unique values
 Values : ['No' 'Yes' 'No internet service']

OnlineBackup:3 unique values
 Values : ['Yes' 'No' 'No internet service']

DeviceProtection:3 unique values
 Values : ['No' 'Yes' 'No internet service']

TechSupport:3 unique values
 Values : ['No' 'Yes' 'No internet service']

StreamingTV:3 unique values
 Values : ['No' 'Yes' 'No internet service']

StreamingMovies:3 unique values
 Values : ['No' 'Yes' 'No internet service']

Contract:3 unique values
 Values : ['Month-to-month' 'One year' 'Two year']

PaperlessBilling:2 unique values
 Values : ['Yes' 'No']

PaymentMethod:4 unique values
 Values : ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
 'Credit card (automatic)']

MonthlyCharges:1585 unique values

TotalCharges:6531 unique values

Churn:2 unique values
 Values : ['No' 'Yes']
'''

# Missing values in TotalCharges after conversion to numeric form: 11
'''
Features Shape: (7043, 19)
Target Shape: (7043,)
Target: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
'''

# Final shape of processed features: (7043, 45)
# Encoded target shape: (7043,)


'''
FINAL PREPROCESSED DATA SUMMARY:
TOTAL FEATURES : 45
All features are now numerical : [dtype('float64') dtype('bool')]
No missing values : True
Target is binary encoded : [0 1]

Data Preprocessing Complete! Ready for algorithm implementation!
'''