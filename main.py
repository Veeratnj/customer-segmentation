import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # To save/load the model

# Load data
df = pd.read_csv('your_dataset.csv')

# Handle missing CustomerID values
df = df.dropna(subset=['CustomerID'])

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Create RFM dataframe
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (df['InvoiceDate'].max() - date.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'UnitPrice': lambda x: (x * df.loc[x.index, 'Quantity']).sum()  # Monetary
})

rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'UnitPrice': 'Monetary'
}, inplace=True)

# Scaling data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])  # Scale only numerical features

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Save the scaler and model to disk
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
joblib.dump(kmeans, 'kmeans_model.pkl')  # Save the KMeans model

# Visualize the clusters (considering performance for large datasets)
sns.set(style="whitegrid")
sns.pairplot(rfm_df[['Recency', 'Frequency', 'Monetary', 'Cluster']], hue='Cluster', palette='Set1', plot_kws={'alpha':0.6})
plt.show()

# Analyze segments by printing the mean values for each cluster
print(rfm_df.groupby('Cluster').mean())
