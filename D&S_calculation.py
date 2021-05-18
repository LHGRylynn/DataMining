import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets

iris=datasets.load_iris()
A=iris.data

print("Iris Dataset:")
print(A)
print()

dist_Euclidean = pdist(A, metric='euclidean')
print("Euclidean:")
print(squareform(dist_Euclidean))
print()

dist_Manhattan = pdist(A, metric='cityblock')
print("Manhattan:")
print(squareform(dist_Manhattan))
print()

dist_Mahalanobis = pdist(A, metric='mahalanobis', VI=None)
print("Mahalanobis:")
print(squareform(dist_Mahalanobis))
print()

dist_Cosine = pdist(A, metric='cosine')
print("Cosine:")
print(squareform(dist_Cosine))
print()

B = np.array([[0, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0]])
print("Binary dataset")
print(B)
print()

dist_Jaccard = pdist(B, metric='jaccard')
print("Jaccard:")
print(squareform(dist_Jaccard))
print()
