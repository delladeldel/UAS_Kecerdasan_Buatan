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
customer_data = pd.read_csv('Mall_Customers.csv')

# Display the data
st.write("### Customer Data", customer_data.head())

# Display the shape of the dataset
st.write("### Data Shape", customer_data.shape)

# Display dataset info
st.write("### Dataset Info")
buffer = io.StringIO()
customer_data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Check for missing values
st.write("### Missing Values", customer_data.isnull().sum())

# Extract features for clustering
X = customer_data.iloc[:, [3, 4]].values

# Finding WCSS for different number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
st.write("### Elbow Point Graph")
plt.figure(figsize=(8, 5))
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# Train the KMeans model with the optimal number of clusters (let's assume 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
Y = kmeans.fit_predict(X)

# Plot the clusters
st.write("### Customer Groups")
plt.figure(figsize=(8, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
st.pyplot(plt)

# Save the trained KMeans model to a file
kmeans_model_filename = 'kmeans_customer_model.sav'
with open(kmeans_model_filename, 'wb') as file:
    pickle.dump(kmeans, file)
st.write(f"KMeans model saved to {kmeans_model_filename}")

# Predicting cluster for new data points
st.write("### Predict Customer Cluster")
annual_income = st.number_input('Annual Income', min_value=0, max_value=150, value=50)
spending_score = st.number_input('Spending Score', min_value=0, max_value=100, value=50)
new_data = np.array([[annual_income, spending_score]])

# Load the trained model
with open(kmeans_model_filename, 'rb') as file:
    loaded_kmeans = pickle.load(file)

# Predict the cluster
cluster = loaded_kmeans.predict(new_data)

st.write(f"The predicted cluster for a customer with annual income {annual_income} and spending score {spending_score} is: Cluster {cluster[0] + 1}")
