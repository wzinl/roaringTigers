import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Step 1: Load the data and clean
file_path = r"dataset\extracted_info_detailed.xlsx"
data = pd.read_excel(file_path)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Drop empty rows and columns
data.dropna(how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)

# Ensure there is a column with text for embeddings
text_column = 'summary'  # Adjust based on the dataset structure
assert text_column in data.columns, f"'{text_column}' column is missing in the dataset."

# Step 2: Generate embeddings
text_data = data[text_column].dropna().tolist()  # Remove empty summaries
embeddings = model.encode(text_data)

# Step 3: Perform clustering
n_clusters = 10  # Define the number of clusters (adjust as needed)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels to the dataframe
data = data.iloc[:len(text_data)]  # Ensure the DataFrame matches the text data size
data['cluster'] = clusters

# Step 4: Reduce dimensions for 3D visualization
pca = PCA(n_components=3)  # Reduce to 3 dimensions
reduced_embeddings = pca.fit_transform(embeddings)

# Step 5: Plot the 3D clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
for cluster_id in range(n_clusters):
    cluster_points = reduced_embeddings[clusters == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
               label=f'Cluster {cluster_id}', s=50)

ax.set_title('3D Visualization of Text Clusters')
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_zlabel('PCA Dimension 3')
ax.legend()
plt.show()
