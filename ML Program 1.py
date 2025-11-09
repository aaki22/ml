# Experiment 1: Structuring Data for ML

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset

data = pd.read_csv("./ML Program Dataset/Iris.csv")

# Step 2: Display the dataset
print("Sample Data:\n", data.head())

# Step 3: Identify features (X) and target (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Step 4: Encode categorical target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded)

# Step 6: Display structured data
print("\nStructured Training Data (X_train):\n", X_train.head())
print("\nTraining Labels (y_train):\n", y_train)

print("\nData Structuring Completed Successfully!")
