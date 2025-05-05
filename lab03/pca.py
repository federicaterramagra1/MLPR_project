
import numpy
import sklearn.datasets 



import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg

def load_data(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            attrs = line.strip().split(',')  # split each line by comma
            features = list(map(float, attrs[:-1]))  # convert feature values to float
            label = int(attrs[-1])  # convert label to int
            DList.append(features)
            labelsList.append(label)

    return numpy.array(DList).T, numpy.array(labelsList)

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

import os

def plot_pca_projection(D, L, P):
    os.makedirs("plots/pca", exist_ok=True)

    mu = D.mean(axis=1).reshape(-1, 1)
    DC = D - mu
    DP = P.T @ DC
    DP[1] *= -1  # inverti y per visualizzazione

    class_names = {0: 'Fake fingerprint', 1: 'Genuine fingerprint'}

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.scatter(DP[0, L == label], DP[1, L == label], label=class_names[label])
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Projection (First 2 Components)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/pca/pca_projection_2D.png")
    plt.close()


def plot_pca_histogram(D, L, P, direction):
    os.makedirs("plots/pca/histograms", exist_ok=True)

    DP = P.T @ D
    DP_dir = DP[direction]

    class_names = {0: 'Fake fingerprint', 1: 'Genuine fingerprint'}
    labels = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(DP_dir[L == label], bins=50, alpha=0.5, label=class_names[label], density=True)
    
    plt.xlabel(f'{labels[direction]} Principal Component')
    plt.ylabel('Density')
    plt.title(f'Histogram of {labels[direction]} PCA Component')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/pca/histograms/hist_pca_{labels[direction].lower()}.png")
    plt.close()



if __name__ == '__main__':

    D, L = load_data('trainData.txt')
    mu, C = compute_mu_C(D)
    print(mu)
    print(C)
    P = compute_pca(D, m = 6)
    print(P)
  
    
    plot_pca_projection(D, L, P)
    for i in range(6):
        plot_pca_histogram(D, L, P, i)


    