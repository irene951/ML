
# PCA and LDA Dimensionality Reduction Implementation
This code implements two common dimensionality reduction techniques: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA), applying them to the Iris dataset.

## Description

This project aims to:
1.  Implement the core steps of PCA and LDA algorithms.
2.  Reduce the dimensionality of the high-dimensional Iris dataset (4 dimensions) to 2 dimensions.
3.  Visualize the original data as well as the results after PCA and LDA dimensionality reduction to compare the differences between the two methods.

## Dataset

* **Source:** `sklearn.datasets.load_iris`
* **Content:** Contains 150 samples of Iris flowers belonging to 3 different species (Setosa, Versicolour, Virginica), with 50 samples per species.
* **Features:** Each sample includes 4 features (measured in centimeters):
    * Sepal length
    * Sepal width
    * Petal length
    * Petal width

## Methods/Algorithms

### 1. Principal Component Analysis (PCA)
* **Type:** Unsupervised Learning
* **Goal:** Find the directions (principal components) that maximize the variance of the data.
* **Implementation Steps:**
    1.  Standardize the data (zero mean, unit variance).
    2.  Calculate the Covariance Matrix.
    3.  Solve for the Eigenvalues and Eigenvectors of the covariance matrix.
    4.  Select the top *k* eigenvectors corresponding to the largest eigenvalues as the principal components.
    5.  Project the standardized data onto the selected principal components.

### 2. Linear Discriminant Analysis (LDA)
* **Type:** Supervised Learning
* **Goal:** Find the directions that maximize the ratio of **Between-Class Scatter (SB)** to **Within-Class Scatter (SW)** for optimal class separation.
* **Implementation Steps:**
    1.  Calculate the Within-Class Scatter matrix (SW) and the Between-Class Scatter matrix (SB).
    2.  Solve the generalized eigenvalue problem for `inv(SW) * SB`. *(The code uses Tikhonov regularization to handle potential singularity of the SW matrix)*.
    3.  Select the top *c-1* eigenvectors corresponding to the largest eigenvalues as the discriminant components (2 for the Iris dataset).
    4.  Project the original data onto the selected discriminant components.

## Dependencies

* Python 3.x
* numpy
* matplotlib
* pandas
* seaborn
* scikit-learn (`sklearn.datasets`)
* scipy (`scipy.linalg`)


## How to Run

1.  Ensure the required dependencies listed above are installed.
2.  Open the `code_111030021.ipynb` file.
3.  Run all cells in the notebook sequentially.

## Results and Visualization

The notebook generates the following visualizations:

1.  **Original Data Pair Plot:** A scatter plot matrix of the original 4 features using Seaborn, colored by Iris species, to observe the initial feature distributions and relationships between classes.
2.  **PCA Projection Plot:** A 2D scatter plot of the data projected onto the first two principal components, colored by Iris species.
3.  **LDA Projection Plot:** A 2D scatter plot of the data projected onto the first two linear discriminants, colored by Iris species.

Additionally, the last cell of the notebook outputs the top two eigenvalues for PCA and LDA for reporting or analysis purposes:

```
--- Summary Data (for Report) ---
PCA Eigenvalues (Variance): [2.9380850501999958, 0.9201649041624861]
LDA Eigenvalues (Between/Within Ratio): [32.19192574115118, 0.2853910161900479]
```

## Code Structure

The main functions are defined within the notebook:

  * `standardize_data(X)`: Standardizes the data.
  * `calculate_covariance_matrix(X_std)`: Calculates the covariance matrix.
  * `pca(X, n_components)`: Executes the PCA process.
  * `calculate_scatter_matrices(X, y)`: Calculates the SW and SB matrices.
  * `lda(X, y, n_components)`: Executes the LDA process.
  * `plot_reduced_data(X_reduced, y, title)`: Plots the reduced 2D data.

<!-- end list -->
