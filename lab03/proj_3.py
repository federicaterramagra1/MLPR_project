#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:29:56 2024

@author: fede
"""

import numpy as np
import matplotlib.pyplot as plt

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            attrs = line.strip().split(',')  # split each line by comma
            features = list(map(float, attrs[:-1]))  # convert feature values to float
            label = int(attrs[-1])  # convert label to int
            DList.append(features)
            labelsList.append(label)

    return np.array(DList).T, np.array(labelsList)

import os

import os  # serve per gestire directory

def plot_hist(D, L):
    os.makedirs("img_proj3/plots/hist", exist_ok=True)  # crea la directory se non esiste

    class_names = {0: 'Class 0 (False)', 1: 'Class 1 (True)'}
    colors = {0: 'royalblue', 1: 'orange'}

    for dIdx in range(D.shape[0]):
        plt.figure()
        for cls in [0, 1]:
            plt.hist(D[dIdx, L == cls], bins=50, alpha=0.6, density=True,
                     label=class_names[cls], color=colors[cls])
        
        plt.title(f'Histogram - Feature {dIdx}')
        plt.xlabel(f'Feature {dIdx}')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = f'img_proj3/plots/hist/hist_feature{dIdx}_Fingerprint.png'
        plt.savefig(save_path)
    plt.close('all')

def plot_scatter(D, L):
    os.makedirs("img_proj3/plots/scatter", exist_ok=True)

    class_names = {0: 'Class 0 (False)', 1: 'Class 1 (True)'}
    colors = {0: 'royalblue', 1: 'orange'}

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i >= j:
                continue
            plt.figure()
            for cls in [0, 1]:
                plt.scatter(D[i, L == cls], D[j, L == cls], label=class_names[cls],
                            alpha=0.5, s=10, color=colors[cls])
            
            plt.xlabel(f'Feature {i}')
            plt.ylabel(f'Feature {j}')
            plt.title(f'Scatter Plot - Feature {i} vs {j}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = f'img_proj3/plots/scatter/scatter_f{i}_vs_f{j}.png'
            plt.savefig(save_path)
    plt.close('all')


if __name__ == '__main__':
    # Load the dataset
    D, L = load('trainData.txt')

    # Plot histograms and scatter plots
    plot_hist(D, L)
    plot_scatter(D, L)

    # Statistical Analysis
    # Calculate mean, covariance, variance, and standard deviation
    # Analyze the histograms and scatter plots to answer the provided questions.

    mu = D.mean(1).reshape((D.shape[0], 1)) # mean(1) => media sulle colonne per ogni feature
    print('Mean:')                          # e trasformo il vettore delle medie delle features in un 
    print(mu)                               # vettore colonna
    print()

    DC = D - mu                 # we center the data
    plot_hist(DC, L)
    plot_scatter(DC, L)
    
      

    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])     # 4 features => matrice 4x4
    print('Covariance:')                                # (D - mu)  calcola la matrice Dcentered​, dove D è la matrice dei dati e μ è il vettore delle medie delle colonne di DD. Questa operazione sottrae il vettore delle medie da ciascuna colonna di DD, centrando così i dati intorno alla media zero
    print(C)                                             # il prodotto tra la matrice Dcentered​ e la sua trasposta DcenteredT​. Questo prodotto produce una matrice simmetrica che rappresenta la covarianza tra le variabili nel dataset D
    print()                         # Dividendo il risultato per il numero di colonne di D (cioè il numero di campioni) con float(D.shape[1])float(D.shape[1])
                                    # si ottiene la matrice di covarianza empirica normalizzata per il numero di osservazioni. Questa normalizzazione rende la matrice di covarianza dipendente solo dalle caratteristiche dei dati e non dalla loro dimensione

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
    
    for cls in [0,1]:
        print('Class', cls)
        DCls = D[:, L==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)     #media lungo le colonne mean(1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)           # varianza lungo l'asse delle colonne, cioè calcola la varianza per ciascuna riga dell'array
        std = DCls.std(1)           # deviazione standard per ciascuna riga dell'array
        print('Variance:', var)
        print('Std. dev.:', std)
        print()