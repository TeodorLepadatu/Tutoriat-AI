# Kernel methods

Think about the following situation: we have 2 classes that we want to separate using a single line. The classical example is the XOR problem, where the 2 classes (0 and 1) can't be seperable using a single. A liniar classification method will not be able to learn the XOR function.

Suppose that in a subspace $R^m$, the data is not linearly separable. **How do we fix this?** We try to apply a function ($f:R^m \rightarrow R^n, m<n$) to the data, such that the data in the subspace $R^n$ is linearly separable.

For example, consider a $x_i \in R^d, d<5$. We apply a function $\phi : R^d \rightarrow R^5 , \phi (x_i) = (1, {x_i}^2, x_i[1] * x_i[2] - x_i[3], \cos(x_i), e^{x_i})$. In this $R^5$ subspace, the data is seperable and we can use a linear separable. 

So:

- we have an $x \in R^d$
- we transform $x$ in $\phi (x) = (\phi_1(x), ..., \phi_D(x))$, where D is larger than d
- then $\phi : R^d \rightarrow R^D$ is called *feature map*
- nothing happens to the labels (y)
- our linear function $f_w(x) = w^T \phi(x) = \sum_{j=1}^D w_j \phi_j(x)$, where $W$ is the matrix for the linear classification $y = Wx$ and it needs to be calculated (or learned)
