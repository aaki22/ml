import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load dataset
url = "./ML Program Dataset/Titanic-Dataset.csv"
data = pd.read_csv(url)

# Step 2: Handle missing values (safe way)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Step 3: Encode categorical columns
label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])

# Step 4: Scale numeric features
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Step 5: Display processed data
print(data[['Sex', 'Age', 'Fare', 'Embarked']].head())
