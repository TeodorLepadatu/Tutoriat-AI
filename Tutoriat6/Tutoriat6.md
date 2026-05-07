# Kernel Methods

Think about the following situation: we have 2 classes that we want to separate using a single line. The classical example is the XOR problem, where the 2 classes (0 and 1) can't be separable using a single line. A linear classification method will not be able to learn the XOR function.

Suppose that in a subspace $\mathbb{R}^m$, the data is not linearly separable. **How do we fix this?** We try to apply a function ($f: \mathbb{R}^m \rightarrow \mathbb{R}^n, m<n$) to the data, such that the data in the subspace $\mathbb{R}^n$ is linearly separable.

For example, consider an $x_i \in \mathbb{R}^d, d<5$. We apply a function $\phi : \mathbb{R}^d \rightarrow \mathbb{R}^5, \phi(x_i) = (1, x_i^2, x_i^{(1)} \times x_i^{(2)} - x_i^{(3)}, \cos(x_i), e^{x_i})$. In this $\mathbb{R}^5$ subspace, the data is separable and we can use a linear classifier.

So:
* We have an $x \in \mathbb{R}^d$.
* We transform $x$ into $\phi(x) = [\phi_1(x), ..., \phi_D(x)]^T$, where $D$ is a dimension much larger than $d$.
* The function $\phi : \mathbb{R}^d \rightarrow \mathbb{R}^D$ is called a *feature map*.
* Nothing happens to the labels ($y$).
* Our linear function is $f_w(x) = w^T \phi(x) = \sum_{j=1}^D w_j \phi_j(x)$, where $w \in \mathbb{R}^D$ needs to be calculated.

---

### **The Challenge of High Dimensions and Regularization**

Before using kernels, a standard linear model aims to find weights $w$ by minimizing the loss function alongside a regularization term: $\sum_{i=1}^N (w^T x_i - y_i)^2 + \lambda ||w||_2^2$. The regularization parameter ($\lambda ||w||_2^2$) is crucial because it ensures the solution remains stable; without it, poor matrix conditioning can result in extremely large weight values with opposite signs that nearly cancel each other out. 

However, when we map our data to a much larger dimension $D$ via the feature map $\Phi$, solving for $w$ directly becomes a problem of size $D \times D$. Because $D$ can be enormous—or even infinite—solving this directly is often impossible.

### **The Representation Theorem**

To bypass the massive computational cost of the $D \times D$ matrix, we rely on the **Representation Theorem**. 

* The theorem states that the solution $w$ can be expressed simply as a linear combination of the mapped data points: $w = \Phi c = \sum_{i=1}^N \phi(x_i)c_i$.
* The new coefficients $c$ are computed as $c = (\Phi^T \Phi + \lambda I_N)^{-1}y$.
* This breakthrough shrinks the problem down from an impossible $D \times D$ dimension to an $N \times N$ dimension, where $N$ is simply the number of data points we have.

### **The Kernel Trick**

The most important realization is that we do not actually need to know the explicit mapping function $\phi(x)$. 
* We only need to know how to compute the dot product between two mapped points: $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$.
* This concept is famously known as the **"kernel trick"**. 
* Instead of working with the mapping $\Phi$, we calculate an $N \times N$ Kernel matrix ($K$), where the element at position $(i,j)$ is simply the result of $\phi(x_i)^T \phi(x_j)$. 
* For this to work mathematically, the kernel function $k(x_i, x_j)$ must be symmetric, and the entire matrix $K$ must be symmetric and positive definite (all eigenvalues must be positive).

### **Common Kernel Functions**

There are several standard kernel functions you can use depending on the data:
* **Linear Kernel:** $k(x_i, x_j) = x_i^T x_j$.
* **Polynomial Kernel:** $k(x_i, x_j) = (x_i^T x_j + 1)^p$.
* **Gaussian Kernel:** $k(x_i, x_j) = \exp(-\frac{||x_i - x_j||_2^2}{2\sigma^2})$.
* **Gaussian RBF Kernel:** $k(x_i, x_j) = \exp(-\gamma||x_i - x_j||_2^2)$.
* **Sigmoid Kernel:** $k(x_i, x_j) = \tanh(\alpha x_i x_j + \gamma)$.

### **How to Use Kernel Methods in Practice**

**1. Training:**
* Start with a dataset $S = \{(x_1, y_1), ..., (x_N, y_N)\}$ and choose a kernel function $k(x_i, x_j)$.
* Compute the $N \times N$ kernel matrix $K$.
* Solve the system to find the coefficients: $c = (K + \lambda I_N)^{-1}y$, where the regularization parameter $\lambda \in \mathbb{R}_+$ is chosen by the user.

**2. Testing (Prediction):**
* Take a new data point $x_{N+1}$.
* Calculate the vector $K_x$ by computing $k(x_{N+1}, x_i)$ against every point $i = 1, ..., N$ in the training set.
* The final prediction is generated using the dot product: $f(x_{N+1}) = K_x^T c$.

*Note: Kernel methods operate on a solid theoretical foundation utilizing Hilbert spaces (which deal with infinite-dimensional vectors and matrices) and, alongside Neural Networks, serve as the primary methodology for introducing non-linearity into machine learning problems.*

### Application of Kernel Methods: Support Vector Machines

A Support Vector Machine (SVM) is a powerful supervised learning model primarily used for classification tasks. While it can handle complex, high-dimensional data, its core concept is highly intuitive and geometric.

The primary objective of an SVM is to draw a line (or a multi-dimensional plane) that separates different classes of data. Instead of just finding *any* line that separates the classes, the SVM algorithm looks for the optimal line that provides the maximum space between the classes.

#### Core Concepts

*   **Hyperplane:** In a 2D space, this is simply a straight line that separates two classes (e.g., spam vs. not spam). In 3D, it is a flat plane. For higher dimensions, it is referred to generally as a hyperplane. This is your decision boundary.
*   **Margin:** This is the distance between the hyperplane and the closest data points from each class. The algorithm is designed to maximize this margin, making the model more robust to new, unseen data.
*   **Support Vectors:** These are the critical data points that lie closest to the decision boundary. They are called "support vectors" because they actively define the margin and the position of the hyperplane. If you remove or move other data points further away, the boundary will not change, but moving a support vector will alter the model.
*   **The "Kernel Trick":** As discussed in your kernel methods context, real-world data is rarely separable by a simple straight line. SVMs use kernel functions (like Polynomial or Radial Basis Function) to project the data into a higher-dimensional space where a linear separation becomes possible, without incurring massive computational costs.

---

### Implementation Example

Below is a Python code snippet using `scikit-learn` that solves a binary classification problem using an SVM with a linear kernel. 

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data[:, :2] # take only 2 dimensions for visualization
y = iris.target

# class 0: Setosa, class 1: Versicolor
binary_mask = y != 2
X = X[binary_mask]
y = y[binary_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_classifier = SVC(kernel='linear', C=1.0)

svm_classifier.fit(X_train, y_train)

predictions = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
```
