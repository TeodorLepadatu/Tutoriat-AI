# KNN

K‑Nearest Neighbors (KNN) is a simple machine learning algorithm for classification and regression tasks. It works by identifying the K closest data points to a given input and making predictions based on the majority class (for classification) or average value of those neighbors (for regression).

To use the KNN algorithm we need 2 things:

- a distance or a similarity function (for example Euclidian distance, Manhattan distance, Minkowski distance ($p!=0$), Cosine Similarity, etc)
- the value for $K$, which would be the number of closest neighbors we look at

**How to choose K?** It needs to be an odd number so that we can have a majority. Usually, we want to pick a decently high K if the data has a lot of noise or outliers, but if it too large, then the model will overfit.

### Problem: [2 moons](https://www.researchgate.net/figure/Results-for-the-two-moons-dataset-2000-points-in-100-dimensions_fig1_255570087)

This problem is a binary classification problem in which you have to classify points which are located on 2 moon-like structures. Each point has 2 features: the X coordinate and the Y coordinate.

**Solution:**

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def classify_two_moons():
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    n_neighbors = 5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # when we use KNN, we want to standardize the data so that we can compare distances with similar orders of magnitude
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{n_neighbors}NN Accuracy on Two Moons: {accuracy:.4f}")


if __name__ == "__main__":
    classify_two_moons()
```
The time complexity for each query point is $O(n*d)$, where n is the number of learned points and d is the number of dimensions.

*Remark:* This problem has only 2 dimensions, so the inference is going to be fast. But if the number is dimensions is high, the algorithm will be very slow and also bad in terms of accuracy.

This is called **curse of dimensionality** and we have some ways in which we can fix it, but the main one is trying to reduce the number of dimensions while keeping the same amount of information.

# PCA

The PCA algorithm is the most common way of reducing the number of dimensions for the data while keeping the most important information. It changes complex datasets by transforming correlated features into a smaller set of uncorrelated components.

**How it works:**

- Step 1: Standardize the data (making each feature have a mean of 0 and a standard deviation of 1).
- Step 2: Calculate the covariance matrix for the dataset. This matrix is populated by calculating the covariance between every pair of features (x̄<sub>1</sub> and x̄<sub>2</sub> are the sample mean values of variables x<sub>1</sub> and x<sub>2</sub>, respectively, *n* is the total number of samples, and x<sub>1i</sub> and x<sub>2i</sub> are the individual data points at index *i*). The formula for the covariance between 2 features is the following:

$$\text{cov}(x_1, x_2) = \frac{1}{n-1} \sum_{i=1}^{n} (x_{1i} - \bar{x}_1)(x_{2i} - \bar{x}_2)$$

- Step 3: Calculate the eigenvectors and the eigenvalues of the covariance matrix (for each eigenvector, its eigenvalue represents its importance, or the amount of variance it explains).
- Step 4: Sort the eigenvectors in descending order based on their eigenvalues and select the top *k* eigenvectors to form a projection matrix.
- Step 5: Transform the original dataset by multiplying it by the projection matrix to map the data into the new, lower-dimensional space.

### Problem: [MNIST](https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python)

The MNIST dataset contains a collection of 70,000, 28 x 28 grayscale images of handwritten digits from 0 to 9, which we are going to classify. For each photo we have 28 x 28 features, which are way too many for a KNN classifier. We are going to use PCA to reduce the dimensions and then apply KNN.

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


def evaluate_mnist_comparison():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    X = mnist.data[:70000]
    y = mnist.target[:70000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # KNN without PCA ---
    start_time_base = time.time()
    knn_base = KNeighborsClassifier(n_neighbors=21)
    knn_base.fit(X_train_scaled, y_train)
    y_pred_base = knn_base.predict(X_test_scaled)
    accuracy_base = accuracy_score(y_test, y_pred_base)
    time_base = time.time() - start_time_base

    # KNN with PCA ---
    start_time_pca = time.time()
    pca = PCA(n_components=128)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn_pca = KNeighborsClassifier(n_neighbors=21)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    time_pca = time.time() - start_time_pca

    components_used = pca.n_components_

    return accuracy_base, time_base, accuracy_pca, time_pca, components_used


if __name__ == "__main__":
    print("Fetching MNIST and evaluating...")
    acc_base, t_base, acc_pca, t_pca, n_comp = evaluate_mnist_comparison()

    print(f"\n--- KNN without PCA (784 features) ---")
    print(f"Accuracy: {acc_base * 100:.2f}%")
    print(f"Time:     {t_base:.2f} seconds")

    print(f"\n--- KNN with PCA ({n_comp} features) ---")
    print(f"Accuracy: {acc_pca * 100:.2f}%")
    print(f"Time:     {t_pca:.2f} seconds")
```

Without PCA, we have 784 features and we got 92.82% accuracy in 4.97 seconds. If we use PCA, we got 94.36% accuracy in 1.65 seconds.

**Exercise:** Use KNN and PCA on the *Iris* dataset. You can find more information about this dataset [here](https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html).
