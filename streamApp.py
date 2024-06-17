import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Title of the application
st.title('Customer Segmentation using K-Means Clustering')

# Load data
@st.cache
def load_data():
    customer_data = pd.read_csv('Mall_Customers.csv')
    return customer_data

customer_data = load_data()

# Display the first few rows
st.write("### Customer Data (First 5 rows)")
st.dataframe(customer_data.head())

# Data shape and info
st.write("### Data Information")
st.write("Shape of the dataset:", customer_data.shape)
st.write("Information of the dataset:")
st.write(customer_data.info())

# Check for missing values
st.write("### Missing Values Check")
st.write(customer_data.isnull().sum())

# Choosing Annual Income and Spending Score columns
X = customer_data.iloc[:, [3, 4]].values

# Calculate WCSS for different number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Point Graph
st.write("### Elbow Point Graph")
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', ax=ax)
ax.set_title('The Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Optimum number of clusters
st.write("### Optimum Number of Clusters")
st.write("From the above elbow graph, we can see that the optimum number of clusters is around 5.")

# Training the k-Means model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

# Visualizing all the clusters
st.write("### Visualizing Clusters")
fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and axis object
ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
ax.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
ax.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
ax.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plot centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

ax.set_title('Customer Groups')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.legend()

# Show plot using st.pyplot()
st.pyplot(fig)
