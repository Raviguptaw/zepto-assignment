import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Step 1: Data Preparation
# Merge customer and transaction data
merged_data = pd.merge(transactions, customers, on="CustomerID", how="inner")

# Step 2: Feature Engineering
# Create features for clustering
customer_features = merged_data.groupby("CustomerID").agg(
    total_spent=pd.NamedAgg(column="TotalValue", aggfunc="sum"),
    avg_spent=pd.NamedAgg(column="TotalValue", aggfunc="mean"),
    transaction_count=pd.NamedAgg(column="TransactionID", aggfunc="count"),
    avg_quantity=pd.NamedAgg(column="Quantity", aggfunc="mean")
).reset_index()

# Add region data to features
def map_region(region):
    mapping = {"Asia": 1, "Europe": 2, "America": 3, "Africa": 4}
    return mapping.get(region, 0)

customer_features = customer_features.merge(customers[["CustomerID", "Region"]], on="CustomerID")
customer_features["Region"] = customer_features["Region"].apply(map_region)

# Step 3: Data Standardization
features = customer_features.drop(columns="CustomerID")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Clustering and Evaluation
results = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    db_index = davies_bouldin_score(scaled_features, labels)
    silhouette_avg = silhouette_score(scaled_features, labels)
    results.append((n_clusters, db_index, silhouette_avg))

# Determine optimal cluster count based on DB Index
optimal_clusters = min(results, key=lambda x: x[1])[0]

# Final Clustering with Optimal Number of Clusters
kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42)
final_labels = kmeans_final.fit_predict(scaled_features)
customer_features["Cluster"] = final_labels

# Step 5: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_features[:, 0], y=scaled_features[:, 1], hue=final_labels, palette="tab10", s=100
)
plt.title("Clusters Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Cluster")
plt.show()

# Print Results
print("Clustering Results:")
for n_clusters, db_index, silhouette_avg in results:
    print(f"Clusters: {n_clusters}, DB Index: {db_index:.2f}, Silhouette Score: {silhouette_avg:.2f}")

print(f"Optimal Number of Clusters: {optimal_clusters}")