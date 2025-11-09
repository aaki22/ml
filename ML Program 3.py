from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ------------------------------
# 1️⃣ Filter Method - SelectKBest
# ------------------------------
filter_selector = SelectKBest(score_func=f_classif, k=2)
filter_selector.fit(X, y)
filter_features = X.columns[filter_selector.get_support()]
print("Filter Method Selected Features:", list(filter_features))

# ------------------------------
# 2️⃣ Embedded Method - Random Forest
# ------------------------------
model = RandomForestClassifier()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
embedded_features = importances.sort_values(ascending=False).head(2).index
print("Embedded Method Selected Features:", list(embedded_features))
