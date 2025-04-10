import pandas as pd

# Load the original CSV
df = pd.read_csv('original.csv')

# Define the cluster to super cluster mapping
mapping = {
    0: 'SC0',
    28: 'SC0',
    47: 'SC1',
    27: 'SC2',
    23: 'SC3',
    48: 'SC3',
    39: 'SC3',
    19: 'SC4',
    22: 'SC4',
    34: 'SC4',
    36: 'SC4',
    29: 'SC4',
    43: 'SC4',
    32: 'SC4'
}

# Create the super_cluster column
df['super_cluster'] = df['cluster_label'].map(mapping)
df['super_cluster'] = df['super_cluster'].fillna('SC5')

# Save to a new CSV
df[['image_path', 'super_cluster']].to_csv('new_super_clusters.csv', index=False)