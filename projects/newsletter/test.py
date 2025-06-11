import faiss
print("FAISS Imported Successfully!")

import faiss
import numpy as np

d = 128  # Vector dimension
nb = 1000  # Number of vectors

np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')

index = faiss.IndexFlatL2(d)  # L2 Distance index
index.add(xb)  # Add vectors to index

print("FAISS indexing test successful!")

