#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fingerprint Spoofing Detection - Lab02
Author: fede
Refactored for GitHub structure and clarity
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    """Load dataset from file and return data matrix (D) and label vector (L)"""
    filename = os.path.join(os.path.dirname(__file__), '..', 'trainData.txt')
    D, L = [], []
    with open(filename) as f:
        for line in f:
            values = line.strip().split(',')
            D.append(list(map(float, values[:-1])))
            L.append(int(values[-1]))
    return np.array(D).T, np.array(L)

def ensure_directories():
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'scatter_plots'), exist_ok=True)

def plot_histograms(D, L, centered=False):
    base_dir = os.path.dirname(__file__)
    prefix = 'hist_centered_' if centered else 'hist_'
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(D.shape[0]):
        plt.figure()
        plt.title(f"Feature {i} {'(centered)' if centered else ''}")
        plt.xlabel(f"Feature {i}")
        plt.hist(D0[i, :], bins=10, density=True, alpha=0.4, label='False')
        plt.hist(D1[i, :], bins=10, density=True, alpha=0.4, label='True')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"histograms/{prefix}{i}.png"))
        plt.close()

def plot_scatter_plots(D, L, centered=False):
    base_dir = os.path.dirname(__file__)
    prefix = 'scatter_centered_' if centered else 'scatter_'
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i == j:
                continue
            plt.figure()
            plt.xlabel(f"Feature {i}")
            plt.ylabel(f"Feature {j}")
            plt.scatter(D0[i, :], D0[j, :], s=1, label='False')
            plt.scatter(D1[i, :], D1[j, :], s=1, label='True')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, f"scatter_plots/{prefix}{i}_{j}.png"))
            plt.close()

def compute_statistics(D, L):
    print("Global statistics")
    mu = D.mean(axis=1, keepdims=True)
    DC = D - mu
    C = DC @ DC.T / D.shape[1]
    print("Mean:\n", mu)
    print("Covariance:\n", C)
    print("Variance:", D.var(axis=1))
    print("Std Dev:", D.std(axis=1), "\n")

    for cls in [0, 1]:
        print(f"Class {cls} statistics")
        D_cls = D[:, L == cls]
        mu_c = D_cls.mean(axis=1, keepdims=True)
        DC_c = D_cls - mu_c
        C_c = DC_c @ DC_c.T / D_cls.shape[1]
        print("Mean:\n", mu_c)
        print("Covariance:\n", C_c)
        print("Variance:", D_cls.var(axis=1))
        print("Std Dev:", D_cls.std(axis=1), "\n")

def main():
    ensure_directories()
    D, L = load_dataset()

    plot_histograms(D, L, centered=False)
    plot_scatter_plots(D, L, centered=False)

    mu = D.mean(axis=1, keepdims=True)
    D_centered = D - mu
    plot_histograms(D_centered, L, centered=True)
    plot_scatter_plots(D_centered, L, centered=True)

    compute_statistics(D, L)

if __name__ == '__main__':
    main()
