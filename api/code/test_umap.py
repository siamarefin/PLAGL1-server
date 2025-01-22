from umap import UMAP
import numpy as np

X = np.random.rand(100, 5)
umap_model = UMAP(n_components=2, random_state=42)
embedding = umap_model.fit_transform(X)

print("UMAP embedding shape:", embedding.shape)
