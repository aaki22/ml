import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv("./ML Program Dataset/winequality-white.csv", sep=";")

# Step 2: Select features (ignore target)
X = data.drop('quality', axis=1)

# Step 3: Scale features
X_scaled = StandardScaler().fit_transform(X)

# Step 4: Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=6, random_state=42)
labels = gmm.fit_predict(X_scaled)

# Step 5: Evaluate clustering
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Step 6: Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=30)
plt.title("GMM Clustering on Wine Dataset")
plt.show()
