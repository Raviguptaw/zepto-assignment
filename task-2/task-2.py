import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Step 1: Data Preparation
# Merge customer and transaction data
merged_data = pd.merge(transactions, customers, on="CustomerID", how="inner")

# Step 2: Feature Engineering
# Create features based on customer and transaction data
customer_features = merged_data.groupby("CustomerID").agg(
    total_spent=pd.NamedAgg(column="TotalValue", aggfunc="sum"),
    avg_spent=pd.NamedAgg(column="TotalValue", aggfunc="mean"),
    transaction_count=pd.NamedAgg(column="TransactionID", aggfunc="count"),
    avg_quantity=pd.NamedAgg(column="Quantity", aggfunc="mean")
).reset_index()

# Add region information to the features
customer_features = customer_features.merge(customers[["CustomerID", "Region"]], on="CustomerID")

# Encode region as numeric
region_mapping = {"Asia": 1, "Europe": 2, "America": 3, "Africa": 4}
customer_features["Region"] = customer_features["Region"].map(region_mapping)

# Step 3: Data Standardization
# Check and handle missing values
features = customer_features.drop(columns="CustomerID")
if features.isnull().values.any():
    print("Missing values found. Filling NaN with 0.")
    features = features.fillna(0)

# Standardize features for similarity calculations
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Calculate Similarity
# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(scaled_features)

# Step 5: Generate Lookalikes
# Map each customer to their top 3 similar customers
lookalike_map = {}
customer_ids = customer_features["CustomerID"].tolist()

for idx, customer_id in enumerate(customer_ids):
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_lookalikes = [
        (customer_ids[sim[0]], sim[1]) for sim in similarity_scores[1:4]  # Skip self-similarity
    ]
    lookalike_map[customer_id] = top_lookalikes

# Step 6: Filter for First 20 Customers
lookalike_results = {
    cust_id: lookalike_map[cust_id] for cust_id in customer_ids[:20]
}

# Step 7: Save Results
# Convert to the required CSV format
lookalike_list = []
for cust_id, lookalikes in lookalike_results.items():
    for similar_cust, score in lookalikes:
        lookalike_list.append({
            "CustomerID": cust_id,
            "LookalikeID": similar_cust,
            "SimilarityScore": score
        })

lookalike_df = pd.DataFrame(lookalike_list)
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike model completed. Results saved to 'Lookalike.csv'.")