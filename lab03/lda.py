#################################################################

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

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return x.reshape((1, x.size))

def compute_mu_C(D): # Same as in pca script
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    #print(Sb)
    #print(Sw)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sbt = numpy.dot(P1, numpy.dot(Sb, P1.T)) # transformed between class covariance
    U2, s2, _ = numpy.linalg.svd(Sbt)

    P2 = U2[:, 0:m] # m highest eigenvalues
    return numpy.dot(P2.T, P1).T

def apply_lda(U, D):
    return U.T @ D

import os 

def plot_lda_projection(D, L, U):
    os.makedirs("plots/lda", exist_ok=True)

    DU = U.T @ D
    class_names = {0: 'Fake fingerprint', 1: 'Genuine fingerprint'}

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.scatter(DU[0, L == label], DU[1, L == label], label=class_names[label])
    
    plt.xlabel('First Linear Discriminant')
    plt.ylabel('Second Linear Discriminant')
    plt.title('LDA Projection (First 2 Directions)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/lda/lda_projection_2D.png")
    plt.close()


def plot_lda_histogram(D, L, U):
    os.makedirs("plots/lda/histograms", exist_ok=True)

    DU = U.T @ D
    DU_first = DU[0]
    class_names = {0: 'Fake fingerprint', 1: 'Genuine fingerprint'}

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(DU_first[L == label], bins=50, alpha=0.5, label=class_names[label], density=True)
    
    plt.xlabel('First LDA Direction')
    plt.ylabel('Density')
    plt.title('Histogram of First LDA Direction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/lda/histograms/hist_lda_first_direction.png")
    plt.close()



if __name__ == '__main__':

    D, L = load_data('trainData.txt')
    U = compute_lda_geig(D, L, m = 2)
    print(U)
    print(compute_lda_JointDiag(D, L, m=2)) # May have different signs for the different directions
    
    U = compute_lda_geig(D, L, m=2)  # Usando la soluzione basata sugli autovettori generalizzati
    #plot_lda_projection(D, L, U)
    plot_lda_histogram(D, L, U)

    