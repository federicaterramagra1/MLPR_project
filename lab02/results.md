# Lab 2 – Fingerprint Spoofing Detection: Full Feature Analysis

In this project, we address a binary classification task: distinguishing between genuine and spoofed fingerprint images using a 6-dimensional feature set. Each data point is labeled as 0 (fake) or 1 (true), and the features are extracted from fingerprint images using a high-level descriptor. The dataset is explored through statistical analysis, histograms, and scatter plots of all feature pairs.

## General Observations

The dataset shows a roughly centered distribution across features, with small global means and unit-scale variances, except for class-dependent variability. Pairwise scatter plots and per-feature histograms reveal varying degrees of class separability and structure.

---

## Analysis of Feature Pairs

### 🔹 Feature 0 and Feature 1

Histograms indicate strong overlap between classes. Feature 0 has nearly identical mean and variance across classes. Feature 1 shows a slight mean shift (class 0 around +0.019, class 1 around –0.008), but still exhibits significant overlap. The scatter plot between these features displays a dense, elliptical cloud with no apparent cluster separation.

**Conclusion**: Low discriminative power, strong overlap, unimodal distributions.

---

### 🔹 Feature 2 and Feature 3

These are the most discriminative features in the dataset. Feature 2 has a symmetric mean shift: class 0 centered around –0.68, class 1 around +0.66. Feature 3 is reversed: class 0 around +0.67, class 1 around –0.66. Both features have very similar variances across classes (~0.55), making the mean difference statistically meaningful.

The scatter plots involving Feature 2 and Feature 3 show vertically or diagonally separated class clusters, with minimal overlap and clear linear decision boundaries.

**Conclusion**: High discriminative potential, well-separated class means, ideal for linear classifiers.

---

### 🔹 Feature 4 and Feature 5

Histograms show mild separation in mean and notable differences in variance: class 1 has variances ~1.3 while class 0 remains around ~0.7. This creates broader distributions for genuine samples.

Scatter plots such as (Feature 4 vs 5) or (Feature 5 vs others) reveal complex clustering patterns. In particular:
- Feature 5 vs Feature 2 shows vertically split blobs.
- Feature 5 vs Feature 3 reveals an upper-lower structure inverted between classes.
- Feature 4 vs Feature 5 shows symmetrical quadrants hinting at double-cluster behavior.

These suggest **multimodal** or **structured intra-class variance**, especially in the genuine class.

**Conclusion**: Moderate class separation, non-linear patterns, visible sub-clusters in scatter plots. Useful when combined with non-linear methods.

---

## Final Summary

- **Best discriminative features**: Feature 2 and Feature 3. They exhibit high inter-class mean separation with similar variance, ideal for LDA or linear SVMs.
- **Weakest features**: Feature 0 and Feature 1, due to strong class overlap and minimal mean difference.
- **Most complex features**: Feature 4 and Feature 5, showing variance-driven spread and multiple visible clusters per class. These may benefit from non-linear modeling or clustering techniques.

The combination of visual and statistical exploration suggests that careful feature selection (e.g., 2 and 3), or projection methods (like LDA or PCA+LDA), will be effective in downstream classification tasks.

