import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
import io

# Set up Streamlit app
st.title('Customer Segmentation using KMeans Clustering')
st.write("""
This app uses KMeans clustering to segment customers based on their annual income and spending score.
""")

# Load the data
data_path = 'Mall_Customers.csv'
customer_data = pd.read_csv(data_path)

# Display the data
st.subheader("Customer Data")
st.dataframe(customer_data)  # This allows for scrolling through the entire dataset

# Display the shape of the dataset
st.subheader("Data Shape")
st.write(f"The dataset contains {customer_data.shape[0]} rows and {customer_data.shape[1]} columns.")

# Display dataset info
st.subheader("Dataset Info")
buffer = io.StringIO()
customer_data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Check for missing values
st.subheader("Missing Values")
st.write(customer_data.isnull().sum())

# Extract features for clustering
X = customer_data.iloc[:, [3, 4]].values

# Finding WCSS for different number of clusters
st.subheader("Elbow Method for Optimal Clusters")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
fig, ax = plt.subplots(figsize=(10, 6))
sns.set()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title('The Elbow Point Graph')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Train the KMeans model with the optimal number of clusters (let's assume 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
Y = kmeans.fit_predict(X)

# Plot the clusters
st.subheader("Customer Groups")
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green', 'red', 'yellow', 'violet', 'blue']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
for i in range(5):
    ax.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=labels[i])

# Plot the centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
ax.set_title('Customer Groups')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.legend()
st.pyplot(fig)

# Save the trained KMeans model to a file
kmeans_model_filename = 'kmeans_customer_model.sav'
with open(kmeans_model_filename, 'wb') as file:
    pickle.dump(kmeans, file)
st.write(f"KMeans model saved to {kmeans_model_filename}")

# Display descriptions of each cluster
st.subheader("Cluster Descriptions")
for i in range(5):
    st.write(cluster_descriptions[i])

# Predicting cluster for new data points
st.subheader("Predict Customer Cluster")
st.write("Enter the annual income and spending score of a customer to predict their cluster.")
annual_income = st.number_input('Annual Income', min_value=0, max_value=150, value=50)
spending_score = st.number_input('Spending Score', min_value=0, max_value=100, value=50)
new_data = np.array([[annual_income, spending_score]])

# Load the trained model
with open(kmeans_model_filename, 'rb') as file:
    loaded_kmeans = pickle.load(file)

# Predict the cluster
cluster = loaded_kmeans.predict(new_data)
predicted_cluster_description = cluster_descriptions[cluster[0]]

st.write(f"The predicted cluster for a customer with annual income {annual_income} and spending score {spending_score} is: Cluster {cluster[0] + 1}")
st.write(predicted_cluster_description)
