# wine datasets + PCA/LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = datasets.load_wine()
X = data['data']
y = data['target']

# Select three features to view the data distribution
def three_features():
    ax = Axes3D(plt.figure())
    for c, i, target_name in zip('>o*', [0, 1, 2], data.target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2],
                   marker=c, label=target_name)
    ax.set_xlabel(data.feature_names[0])
    ax.set_ylabel(data.feature_names[1])
    ax.set_zlabel(data.feature_names[2])
    ax.set_title("wine")
    plt.legend()
    plt.show()

# Select two features to view the data distribution
def two_features():
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title("wine")
    plt.legend()
    plt.show()

# Reduced to two dimensions by PCA
def two_PCA():
    pca = PCA(n_components=2)
    X_p = pca.fit(X).transform(X)
    print(pca.explained_variance_ratio_)  #输出贡献率
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("wine")
    plt.legend()
    plt.show()

# Reduced to two dimensions by PCA after standardization
def standard_two_PCA():
    Xx=StandardScaler().fit(X).transform(X)
    pca_s = PCA(n_components=2)
    X_p =pca_s.fit(Xx).transform(Xx)

    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("wine-standard-PCA")
    plt.legend()
    plt.show()

# Reduced to two dimensions by LDA
def two_LDA():
    lda = LDA(n_components=2)
    X_r =lda.fit(X,y).transform(X)
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("LDA")
    plt.legend()
    plt.show()


def main():
    three_features()   # Select three features to view the data distribution
    two_features()   # Select two features to view the data distribution
    two_PCA()   # Reduced to two dimensions by PCA
    standard_two_PCA()  # Reduced to two dimensions by PCA after standardization
    two_LDA()  # Reduced to two dimensions by LDA


if __name__ == '__main__':
    main()
