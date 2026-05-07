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

### Exercises

**Exercise 1**
Given two vectors $x = [2, 3, 4]^T$ and $y = [3, 4, 5]^T$. Consider a kernel $k(x,y) = (x^T y)^2$ which has the associated transformation $\phi(x) = [x_1^2, x_1 x_2, x_1 x_3, x_2 x_1, x_2^2, x_2 x_3, x_3 x_1, x_3 x_2, x_3^2]^T$. Verify that $k(x,y) = \phi(x)^T \phi(y)$.

**Exercise 2**
Given a polynomial kernel $k(x,y) = (x^T y + 2)^2$ for any $x, y \in \mathbb{R}^2$, calculate the mapping $\phi(z)$. Is this mapping unique?

**Exercise 3**
For the 1-dimensional Radial Basis Function (RBF) kernel, show that the feature space is infinite.

**Exercise 4**
Solve the non-linear XOR classification problem using a polynomial kernel.

---

### Solutions

**Solution 1**
* Our initial vectors contain 3 elements, meaning the initial dimension is $d=3$.
* The transformation mapping $\phi$ results in a vector with 9 elements, meaning the new dimension is $D=9$.
* First, we apply the transformation element by element to vector $x$. Given $x_1=2, x_2=3, x_3=4$:
    $\phi(x) = [2^2, 2 \times 3, 2 \times 4, 3 \times 2, 3^2, 3 \times 4, 4 \times 2, 4 \times 3, 4^2]^T$
    $\phi(x) = [4, 6, 8, 6, 9, 12, 8, 12, 16]^T$
* Next, we apply the same transformation to vector $y$. Given $y_1=3, y_2=4, y_3=5$:
    $\phi(y) = [3^2, 3 \times 4, 3 \times 5, 4 \times 3, 4^2, 4 \times 5, 5 \times 3, 5 \times 4, 5^2]^T$
    $\phi(y) = [9, 12, 15, 12, 16, 20, 15, 20, 25]^T$
* Now, we calculate the dot product explicitly in the 9-dimensional feature space by multiplying corresponding elements and summing them:
    $\phi(x)^T \phi(y) = (4 \times 9) + (6 \times 12) + (8 \times 15) + (6 \times 12) + (9 \times 16) + (12 \times 20) + (8 \times 15) + (12 \times 20) + (16 \times 25)$
    $\phi(x)^T \phi(y) = 36 + 72 + 120 + 72 + 144 + 240 + 120 + 240 + 400$
    $\phi(x)^T \phi(y) = 1444$
* Then, we calculate the kernel directly in the original 3-dimensional space using the definition $k(x,y) = (x^T y)^2$:
    $x^T y = (2 \times 3) + (3 \times 4) + (4 \times 5) = 6 + 12 + 20 = 38$
    $k(x,y) = 38^2 = 1444$
* **Conclusion:** Both methods yield identical results (1444), mathematically validating that $\phi(x)^T \phi(y) = k(x,y)$.

**Solution 2**
* We begin by expanding the kernel function algebraically. Let $x = [x_1, x_2]^T$ and $y = [y_1, y_2]^T$:

$$
k(x,y) = (x^T y + 2)^2 = \left( \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} + 2 \right)^2 = (x_1 y_1 + x_2 y_2 + 2)^2
$$

* We expand the squared polynomial using the formula $(a+b+c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc$:

$$
(x_1 y_1 + x_2 y_2 + 2)^2 = x_1^2 y_1^2 + x_2^2 y_2^2 + 4 + 2 x_1 x_2 y_1 y_2 + 4 x_1 y_1 + 4 x_2 y_2
$$

* To express this sum as a dot product of the form $\phi(x)^T \phi(y)$, we group the corresponding terms for $x$ and $y$ and split the constants symmetrically:
    $= (2 \times 2) + (2x_1 \times 2y_1) + (2x_2 \times 2y_2) + (\sqrt{2}x_1 x_2 \times \sqrt{2}y_1 y_2) + (x_1^2 \times y_1^2) + (x_2^2 \times y_2^2)$
* Extracting the terms that belong solely to the input variable yields the feature map:
    $\phi(z) = [2, 2z_1, 2z_2, \sqrt{2}z_1 z_2, z_1^2, z_2^2]^T$
* **Uniqueness:** This mapping is not strictly unique. One could reorder the elements in the vector, or factor the coefficients differently (e.g., introducing negative signs such as $[-2, -2z_1, -2z_2, -\sqrt{2}z_1 z_2, -z_1^2, -z_2^2]^T$). As long as the dot product $\phi(x)^T \phi(y)$ reconstructs the original expanded polynomial, the mapping is valid.

**Solution 3**
* We use the Taylor series expansion to prove the infinite dimensionality. We start with the 1D RBF kernel formula:

$$
k(x,y) = \exp(-\gamma(x-y)^2) = \exp(-\gamma(x^2 - 2xy + y^2))
$$

* Distribute the $-\gamma$.

$$
\exp(-\gamma x^2 - \gamma y^2 + 2\gamma xy) = \exp(-\gamma x^2) \exp(-\gamma y^2) \exp(2\gamma xy)
$$

* Now, we expand the term $\exp(2\gamma xy)$ using the standard infinite Taylor series $\exp(z) = 1 + \frac{z}{1!} + \frac{z^2}{2!} + \frac{z^3}{3!} + \dots$, where $z = 2\gamma xy$:

$$
= \exp(-\gamma x^2) \exp(-\gamma y^2) \left( 1 + \frac{2\gamma xy}{1!} + \frac{(2\gamma xy)^2}{2!} + \frac{(2\gamma xy)^3}{3!} + \dots \right)
$$

* We then distribute the exponents over $x$ and $y$ to perfectly separate the terms:

$$
= \exp(-\gamma x^2) \exp(-\gamma y^2) \left( 1 + \frac{2\gamma}{1!} x y + \frac{(2\gamma)^2}{2!} x^2 y^2 + \frac{(2\gamma)^3}{3!} x^3 y^3 + \dots \right)
$$

* To construct the dot product format $\phi(x)^T \phi(y)$, we split the coefficients equally using square roots:

$$
= \exp(-\gamma x^2) \exp(-\gamma y^2) \left[ (1 \times 1) + \left(\sqrt{\frac{2\gamma}{1!}}x \times \sqrt{\frac{2\gamma}{1!}}y\right) + \left(\sqrt{\frac{(2\gamma)^2}{2!}}x^2 \times \sqrt{\frac{(2\gamma)^2}{2!}}y^2\right) + \dots \right]
$$

* This allows us to extract the specific mapping vector $\phi(z)$. We pull the $\exp(-\gamma z^2)$ term inside the vector as a scalar multiplier:
    $\phi(z) = \exp(-\gamma z^2) \left[ 1, \sqrt{\frac{2\gamma}{1!}}z, \sqrt{\frac{(2\gamma)^2}{2!}}z^2, \sqrt{\frac{(2\gamma)^3}{3!}}z^3, \dots \right]^T$
* **Conclusion:** Because the Taylor series is an infinite sum, the resulting feature vector $\phi(z)$ inherently contains an infinite number of terms, proving that the RBF feature space is infinite-dimensional.

**Solution 4**
* In the standard XOR problem, data is not linearly separable in 2D space. Instead of using binary 0 and 1, we map the coordinates to -1 and 1. The XOR logic dictates that identical inputs give one class, and opposite inputs give the other class. The four data points and their labels $y_i$ are:
    $x_1 = [-1, -1]^T \implies y_1 = 1$
    $x_2 = [-1, 1]^T \implies y_2 = -1$
    $x_3 = [1, -1]^T \implies y_3 = -1$
    $x_4 = [1, 1]^T \implies y_4 = 1$
* We utilize the polynomial kernel defined as $k(x,y) = (x^T y + 1)^2$. Expanding this algebraically yields the corresponding transformation $\phi(z)$ into a $D=6$ dimensional space:
    $\phi(z) = [1, \sqrt{2}z_1, \sqrt{2}z_2, z_1^2, z_2^2, \sqrt{2}z_1 z_2]^T$
* We map each of our four original points into this 6-dimensional space by substituting their $x_1$ and $x_2$ values:
    $\phi(x_1) = [1, \sqrt{2}(-1), \sqrt{2}(-1), (-1)^2, (-1)^2, \sqrt{2}(-1)(-1)]^T = [1, -\sqrt{2}, -\sqrt{2}, 1, 1, \sqrt{2}]^T$
    $\phi(x_2) = [1, \sqrt{2}(-1), \sqrt{2}(1), (-1)^2, 1^2, \sqrt{2}(-1)(1)]^T = [1, -\sqrt{2}, \sqrt{2}, 1, 1, -\sqrt{2}]^T$
    $\phi(x_3) = [1, \sqrt{2}(1), \sqrt{2}(-1), 1^2, (-1)^2, \sqrt{2}(1)(-1)]^T = [1, \sqrt{2}, -\sqrt{2}, 1, 1, -\sqrt{2}]^T$
    $\phi(x_4) = [1, \sqrt{2}(1), \sqrt{2}(1), 1^2, 1^2, \sqrt{2}(1)(1)]^T = [1, \sqrt{2}, \sqrt{2}, 1, 1, \sqrt{2}]^T$
* To determine if a linear boundary exists in this new space $D=6$ that separates the classes, we must find a linear weight vector $w = [w_1, w_2, w_3, w_4, w_5, w_6]^T$ such that:
    $w^T \phi(x_i) > 0$ when $y_i = 1$
    $w^T \phi(x_i) < 0$ when $y_i = -1$
* To solve this as a single system of inequalities, we multiply the equations for the negative classes by -1, forcing all conditions to be strictly greater than zero ($y_i \cdot w^T \phi(x_i) > 0$):
    For $x_1$ (Class  1): $1 \cdot [1, -\sqrt{2}, -\sqrt{2}, 1, 1, \sqrt{2}] w > 0$
    For $x_2$ (Class -1): $-1 \cdot [1, -\sqrt{2}, \sqrt{2}, 1, 1, -\sqrt{2}] w > 0 \implies [-1, \sqrt{2}, -\sqrt{2}, -1, -1, \sqrt{2}] w > 0$
    For $x_3$ (Class -1): $-1 \cdot [1, \sqrt{2}, -\sqrt{2}, 1, 1, -\sqrt{2}] w > 0 \implies [-1, -\sqrt{2}, \sqrt{2}, -1, -1, \sqrt{2}] w > 0$
    For $x_4$ (Class  1): $1 \cdot [1, \sqrt{2}, \sqrt{2}, 1, 1, \sqrt{2}] w > 0$
* Representing this as a matrix multiplication inequality:

$$
\begin{bmatrix} 
1 & -\sqrt{2} & -\sqrt{2} & 1 & 1 & \sqrt{2} \\ 
-1 & \sqrt{2} & -\sqrt{2} & -1 & -1 & \sqrt{2} \\ 
-1 & -\sqrt{2} & \sqrt{2} & -1 & -1 & \sqrt{2} \\ 
1 & \sqrt{2} & \sqrt{2} & 1 & 1 & \sqrt{2} 
\end{bmatrix} 
\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\ w_5 \\ w_6 \end{bmatrix} > \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

* Inspecting the final column of the matrix, we observe that the value is consistently positive ($\sqrt{2}$) across all rows, representing the feature $\sqrt{2}x_1 x_2$.
* If we select the weight vector $w = [0, 0, 0, 0, 0, 1]^T$, the matrix multiplication simplifies to outputting the last column:

$$
\begin{bmatrix} \sqrt{2} \\ \sqrt{2} \\ \sqrt{2} \\ \sqrt{2} \end{bmatrix} > \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

* **Conclusion:** Because $\sqrt{2} > 0$ holds true for all four points, the chosen weight vector perfectly separates the classes. This proves mathematically that applying the polynomial feature map makes the previously inseparable XOR problem linearly separable in the higher-dimensional space.
