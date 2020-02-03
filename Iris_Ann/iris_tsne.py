import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()

x = iris.data
y = iris.target

# TSNE
x_tsne = TSNE(n_components=2, 
            random_state=42).fit_transform(x)

# Collect output
out = np.zeros((x_tsne.shape[0], 3))
out[:,:2] = x_tsne
out[:,2] = y

# Plot Data
labels = [0, 1, 2]
for lab in labels:
    arr = out[out[:,2] == lab]   
    plt.scatter(arr[:,0], arr[:,1], label=lab)

# Show plot
plt.legend()
plt.show()

   

