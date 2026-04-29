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
