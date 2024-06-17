import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache
def load_data():
    return pd.read_csv('Mall_Customers.csv')

# Main title and description
st.title('Customer Segmentation and Prediction App')
st.write('This app performs customer segmentation using K-Means clustering and saves a RandomForestClassifier model.')

# Load data
customer_data = load_data()

# Display first 5 rows of the dataframe
st.subheader('First 5 rows of the dataset:')
st.write(customer_data.head())

# Display dataset info
st.subheader('Dataset information:')
st.write(customer_data.info())

# Check for missing values
st.subheader('Missing values:')
st.write(customer_data.isnull().sum())

# Perform clustering
st.subheader('Customer Segmentation using K-Means Clustering:')
X = customer_data.iloc[:, [3, 4]].values

# Finding the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
st.subheader('Elbow Point Graph:')
fig = plt.figure()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(fig)

# Perform K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

# Plot clusters and centroids
st.subheader('Customer Groups:')
fig = plt.figure(figsize=(8, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
st.pyplot(fig)

# Training a RandomForestClassifier and saving the model
st.subheader('Training RandomForestClassifier:')
X_train = [[0, 0], [1, 1]]
y_train = [0, 1]
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save the trained model to a file
filename = 'customer_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)

st.write(f"Model saved to {filename}")
