import umap.umap_ as umap  # Correct import

# Initialize UMAP model
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

print("UMAP imported successfully!3")
