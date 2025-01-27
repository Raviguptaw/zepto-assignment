import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
try:
    customers = pd.read_csv("Customers.csv")
    products = pd.read_csv("Products.csv")
    transactions = pd.read_csv("Transactions.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise
except pd.errors.EmptyDataError as e:
    print(f"Error: {e}")
    raise
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise

# Inspect the data
print("Customers Data:")
print(customers.info())
print(customers.head())

print("\nProducts Data:")
print(products.info())
print(products.head())

print("\nTransactions Data:")
print(transactions.info())
print(transactions.head())

# Check for missing values
print("\nMissing values in Customers:")
print(customers.isnull().sum())

print("\nMissing values in Products:")
print(products.isnull().sum())

print("\nMissing values in Transactions:")
print(transactions.isnull().sum())

# Merge datasets for comprehensive analysis
merged_data = pd.merge(transactions, customers, on="CustomerID", how="left")
merged_data = pd.merge(merged_data, products, on="ProductID", how="left")

# Convert date columns to datetime
def convert_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col])

convert_to_datetime(customers, "SignupDate")
convert_to_datetime(transactions, "TransactionDate")

# Basic statistics
print("\nSummary statistics of transaction values:")
print(merged_data["TotalValue"].describe())

# Visualize key trends
sns.histplot(merged_data["TotalValue"], bins=30, kde=True)
plt.title("Distribution of Total Transaction Values")
plt.xlabel("Transaction Value")
plt.ylabel("Frequency")
plt.show()

# Grouped analysis
signup_trends = customers.groupby(customers.SignupDate.dt.year).size()
signup_trends.plot(kind="bar", title="Customer Signups by Year")
plt.xlabel("Year")
plt.ylabel("Number of Signups")
plt.show()

# Popular products
popular_products = merged_data.groupby("ProductName")["Quantity"].sum().sort_values(ascending=False)
print("\nTop 10 Popular Products:")
print(popular_products.head(10))

# Business insights (to be added in the report)
insights = [
    "The distribution of transaction values shows most transactions are under a certain threshold, indicating price-sensitive customers.",
    "Customer signups peaked in certain years, indicating successful marketing or seasonal trends.",
    "Product A (replace with actual product) is the most purchased item, suggesting a high demand for this category.",
    "Certain regions (analyze 'Region') contribute more to transactions, signaling potential focus areas for marketing.",
    "High-value transactions are relatively rare but significantly impact total revenue, indicating the importance of premium products."
]

for idx, insight in enumerate(insights, 1):
    print(f"Insight {idx}: {insight}")
