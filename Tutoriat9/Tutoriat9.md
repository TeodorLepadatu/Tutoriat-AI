# Neural Networks (Multi-Layer Perceptrons)

A neural network is a machine learning model that stacks simple "neurons" in layers and learns pattern-recognizing weights and biases from data to map inputs to outputs. Neural networks are among the most influential algorithms in modern machine learning and artificial intelligence. They are trying to mimic the way the human brain processes information, albeit in a very simplified way.

## Neuron/Perceptron

The very basic building block of a neural network is the neuron, aka perceptron.

A single neuron operates in two distinct phases: calculating a weighted sum of its inputs and applying an activation function to determine the final output.

Each input connection to the neuron has an associated **weight**, which represents the strength or importance of that specific input. The neuron also contains a **bias** term, which acts as an offset, allowing the model to shift the activation threshold independently of the input values.

Given $n$ inputs denoted as $x_1, x_2, \dots, x_n$, their corresponding weights $w_1, w_2, \dots, w_n$, and a bias $b$, the neuron first computes the weighted sum, $z$:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

In linear algebra notation, this is more efficiently expressed as the dot product of the weight vector $\mathbf{w}$ and the input vector $\mathbf{x}$:

$$z = \mathbf{w}^T \mathbf{x} + b$$

If we were to keep the output of the neuron like this, if we were to add more neurons, we would just be doing more linear transformations, which would be completely equivalent to just having one neuron. To make the model more powerful, we need to add a non-linearity to the output of the neuron. This is done by applying an **activation function** to the weighted sum $z$.

**Common Activation Functions**

Depending on the architecture and the specific layer of the network, different mathematical functions are used for $f(z)$:

* **Step Function (Heaviside):** Used in the classical perceptron. It outputs 1 if the weighted sum is positive, and 0 otherwise.

  $$f(z) = 1 \text{, if } z \ge 0$$
  
  $$f(z) = 0 \text{, if } z < 0$$

* **Sigmoid:** Maps the output to a smooth continuous curve between 0 and 1. It is historically significant and often used in binary classification output layers.

  $$f(z) = \frac{1}{1 + e^{-z}}$$

* **Tanh (Hyperbolic Tangent):** Similar to sigmoid but maps the output to a range between -1 and 1, which can be beneficial for certain types of data.

  $$f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

* **ReLU (Rectified Linear Unit):** Outputs the input directly if it is positive, and outputs zero if it is negative. Due to its computational efficiency and resistance to the vanishing gradient problem, it is the standard activation function for hidden layers in modern networks.

  $$f(z) = \max(0, z)$$

* **PReLU:** A variant of ReLU that allows a small, non-zero gradient when the input is negative, which can help mitigate the "dying ReLU" problem.

  $$f(z) = z \text{, if } z > 0$$
  
  $$f(z) = \alpha z \text{, if } z \le 0$$

  where $\alpha$ is a small constant.

* **Leaky ReLU:** Similar to PReLU, it allows a small, non-zero gradient when the input is negative, but with a fixed slope.

  $$f(z) = z \text{, if } z > 0$$
  
  $$f(z) = 0.01z \text{, if } z \le 0$$

* **ELU:** Exponential Linear Unit, which smooths the output for negative inputs and can help with convergence during training.

  $$f(z) = z \text{, if } z > 0$$
  
  $$f(z) = \alpha (e^z - 1) \text{, if } z \le 0$$

  where $\alpha$ is a positive constant (e.g., 1.0).

* **Softmax:** Used in the output layer of multi-class classification problems to convert raw scores (logits) into probabilities.

  $$f(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

  where $z_i$ is the input to the $i$-th neuron in the output layer, and the denominator sums over all neurons in that layer.

* **Maxout:** A generalization of ReLU that takes the maximum of a set of linear functions, allowing for more complex activation patterns.

  $$f(z) = \max_{i} (w_i^T x + b_i)$$

  where $w_i$ and $b_i$ are the weights and biases for the $i$-th linear function.

## Multi-Layer Perceptrons (MLPs)

In practice, we almost never use only one neuron, as they can only learn linear decision boundaries. To learn more complex patterns, we stack multiple layers of neurons on top of each other, creating a multi-layer perceptron (MLP). Each layer transforms the output of the previous layer, allowing the network to learn hierarchical representations of the data.

### The Mathematical Formulation of an MLP

To represent a multi-layer perceptron mathematically, we transition from individual scalar weights to weight matrices and activation vectors. This allows us to express the calculations for an entire layer simultaneously using linear algebra.

Let $L$ be the total number of layers in the network, where layer $0$ is the input layer and layer $L$ is the final output layer. For any given layer $l \in \{1, 2, \dots, L\}$:

* $\mathbf{a}^{[l-1]}$ is the activation vector from the previous layer. For the first hidden layer, this is the input vector ($\mathbf{a}^{[0]} = \mathbf{x}$).
* $\mathbf{W}^{[l]}$ is the weight matrix for layer $l$. Each row contains the weights for a single neuron in that layer.
* $\mathbf{b}^{[l]}$ is the bias vector for layer $l$.
* $f^{[l]}$ is the non-linear activation function for layer $l$.

The feedforward process (forward propagation) for any single layer $l$ is defined by two sequential operations:

**1. The Linear Transformation**
We calculate the weighted sum vector $\mathbf{z}^{[l]}$ for all neurons in the layer:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

**2. The Non-Linear Activation**
We apply the activation function element-wise to the weighted sum vector to get the output vector $\mathbf{a}^{[l]}$ for the current layer:

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

**The Complete Network Formula**

For an $L$-layer network, the final output $\hat{y} = \mathbf{a}^{[L]}$ for a given input $\mathbf{x}$ can be expanded into a single nested formula:

$$\hat{y} = f^{[L]} \Big( \mathbf{W}^{[L]} f^{[L-1]} \big( \dots f^{[1]} ( \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]} ) \dots \big) + \mathbf{b}^{[L]} \Big)$$

Usually, a neural network is very big, with many layers and many neurons, so we can't calculate this by hand or know what matrices and functions to use. So, we need a way to learn the weights and the biases (the activation functions are usually fixed) from the data. 

*Remark:* Usually, we train a network for a specific task, so the weights will be updated based on the training data.

### The training algorithm: Backpropagation

To train a neural network, we are going to use an algorithm called Backpropagation.

The idea of this algorithm is to calculate the gradient of the loss function with respect to each weight and bias in the network, and then update the weights and biases in the direction that minimizes the loss.

### The Mathematics of Backpropagation

Backpropagation relies heavily on the **chain rule** from calculus to propagate the error backwards through the network's layers. The goal is to compute the gradient of the loss function $\mathcal{L}$ with respect to the weight matrices $\mathbf{W}^{[l]}$ and bias vectors $\mathbf{b}^{[l]}$ for every layer $l$.

Before detailing the algorithm, here is the complete mathematical notation used to describe the network and the backpropagation process:

**Notations:**
* $L$: The total number of layers in the neural network (where layer $L$ is the final output layer).
* $l$: The index of a specific layer, ranging from $1$ to $L$.
* $\mathcal{L}$: The loss function, a scalar value representing the difference between predictions and actual targets.
* $\mathbf{a}^{[l]}$: The activation column vector for layer $l$, containing the outputs $a_i^{[l]}$ of all neurons in that layer.
* $\mathbf{y}$: The true target column vector from the dataset.
* $\mathbf{z}^{[l]}$: The weighted sum column vector for layer $l$, containing the pre-activation values $z_i^{[l]}$.
* $\delta^{[l]}$: The error column vector for layer $l$. It represents the gradient of the loss with respect to $\mathbf{z}^{[l]}$.
* $f^{[l]}$: The non-linear activation function used in layer $l$.
* $f'^{[l]}$: The mathematical derivative of the activation function $f^{[l]}$.
* $\mathbf{W}^{[l]}$: The weight matrix connecting layer $l-1$ to layer $l$. A single element $w_{ij}^{[l]}$ connects transmitting neuron $j$ to receiving neuron $i$.
* $\mathbf{b}^{[l]}$: The bias column vector for layer $l$.
* $T$: The transpose operation, which converts a column vector into a row vector (or flips a matrix along its diagonal).
* $\alpha$: The learning rate, a scalar hyperparameter that controls the step size during parameter updates.

**1. The Loss Function and the Output Error**
First, we define a loss function $\mathcal{L}(\mathbf{a}^{[L]}, \mathbf{y})$ that quantifies the difference between the network's final predictions $\mathbf{a}^{[L]}$ and the true targets $\mathbf{y}$.

To begin the backward pass, we calculate the error at the output layer, denoted as $\delta^{[L]}$. This is the gradient of the loss with respect to the weighted sum $\mathbf{z}^{[L]}$. Expanding the element-wise multiplication manually for each neuron $i$ in the output layer:

$$\delta^{[L]} = \begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial a_1^{[L]}} \cdot f'^{[L]}(z_1^{[L]}) \\
\frac{\partial \mathcal{L}}{\partial a_2^{[L]}} \cdot f'^{[L]}(z_2^{[L]}) \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial a_n^{[L]}} \cdot f'^{[L]}(z_n^{[L]})
\end{bmatrix}$$

**2. Propagating the Error Backwards**
For any hidden layer $l$ (where $l$ goes from $L-1$ down to $1$), the error $\delta^{[l]}$ is computed using the error from the subsequent layer $\delta^{[l+1]}$ and the weight matrix $\mathbf{W}^{[l+1]}$ connecting them. Expanding the element-wise multiplication manually for each neuron $k$ in layer $l$:

$$\delta^{[l]} = \begin{bmatrix}
\left( (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \right)_1 \cdot f'^{[l]}(z_1^{[l]}) \\
\left( (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \right)_2 \cdot f'^{[l]}(z_2^{[l]}) \\
\vdots \\
\left( (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \right)_k \cdot f'^{[l]}(z_k^{[l]})
\end{bmatrix}$$

This recursive formula represents the chain rule from calculus, showing exactly how the error distributes itself backward through the network's structure.

**3. Calculating the Gradients**
Once we have the error vectors $\delta^{[l]}$ for all layers, the gradients of the loss with respect to the matrices and vectors are computed:

* **Weight Gradient:** The error column vector multiplied by the transposed activation (row) vector from the previous layer, resulting in a matrix of the same dimensions as $\mathbf{W}^{[l]}$.

  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T$$

* **Bias Gradient:** The gradient with respect to the bias is exactly equal to the error vector.

  $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}$$

**4. The Parameter Update**
Finally, using an optimization algorithm like Gradient Descent (or some variants of it at which we will take a look soon), we update the parameters by taking a step in the opposite direction of the gradient to minimize the loss, scaled by the learning rate $\alpha$:

$$\mathbf{W}^{[l]} = \mathbf{W}^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} = \mathbf{b}^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$$

### Backpropagation Example

Consider a fully connected network with 2 input neurons($x_1$ and $x_2$), one hidden layer with 2 neurons ($h_1$ and $h_2$) and one output layer with 1 neuron ($h_3$).

Suppose the input is $x_1 = 1$, $x_2 = 0$.

$x_1$ is connected to $h_1$ with the weight $w_1 = 1$ and to $h_2$ with the weight $w_2 = 0.5$.

$x_2$ is connected to $h_1$ with the weight $w_3 = 1$ and to $h_2$ with the weights $w_4 = -0.5$.

$h_1$ has a bias $b_1 = 0.5$ and $h_2$ has a bias $b_2 = 0$.

Both $h_1$ and $h_2$ have the activation function ReLU.

$h_1$ is connected to $h_3$ and has a weight $w_5 = 1$.

$h_2$ is connected to $h_3$ and has a weight $w_6 = 1$.

$h_3$ has a bias $b_3 = 0.5$.

The real label of this example is 2. Consider the loss function to be $L = (\hat{y} - y)^2$, where $\hat{y}$ is the output of $h_3$ and $y$ is the real label.

The learning rate of this network is $\eta = 0.01$

Let's do one step of backpropagation on this network.

**Step 1: Forward pass:**

We need to calculate the output of $h_3$ for the given input. So, we have:

$h_1 = \max(0, w_1 x_1 + w_3 x_2 + b_1) = \max(0, 1 \cdot 1 + 1 \cdot 0 + 0.5) = 1.5$
and $h_2 = \max(0, w_2 x_1 + w_4 x_2 + b_2) = \max(0, 0.5 \cdot 1 + 0 \cdot (-0.5) + 0) = 0.5 $

Now, we can calculate the output of $h_3$:

$h_3 = \max(0, w_5 h_1 + w_6 h_2 + b_3) = \max(0, 1 \cdot 1.5 + 1 \cdot 0.5 + 0.5) = 2.5$. The real label is $\hat{y} = 2$, so the loss is $L = (2.5 - 2)^2 = 0.25$.

**Step 2: Backward pass:**

We need to calculate the partial derivatives of the loss with respect to the weights and the biases. We start with the output layer. The loss function is 

$$L = (h_1 w_5 + h_2 w_6 + b_3 - y)^2$$.

The partial derivatives are:

$$\frac{\partial L}{\partial w_5} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) h_1 = 2(2.5 - 2) * 1.5 = 1.5$$

$$\frac{\partial L}{\partial w_6} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) h_2 = 2(2.5 - 2) * 0.5 = 0.5$$

$$\frac{\partial L}{\partial b_3} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) = 2(2.5 - 2) = 1$$

Now, we have to update the weights using gradient descent:

$$w_5 = w_5 - \eta \frac{\partial L}{\partial w_5} = 1 - 0.01 * 1.5 = 0.985$$

$$w_6 = w_6 - \eta \frac{\partial L}{\partial w_6} = 1 - 0.01 * 0.5 = 0.995$$

$$b_3 = b_3 - \eta \frac{\partial L}{\partial b_3} = 0.5 - 0.01 * 1 = 0.49$$

We can now go one step back to the hidden layer. In order to do this correctly, we need to calculate the derivative of the activation function as well. The activation function of $h_1$ and $h_2$ is ReLU and its derivative is:

$$ReLU'(z) = \begin{cases}
1, & \text{if } z > 0 \\
0, & \text{if } z \le 0
\end{cases}$$

We need to calculate the partial derivatives of the loss with respect to $w_1$, $w_3$ and $b_1$, which do not appear explicitly in the loss function, so we need to use the chain rule:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_5 \cdot \frac{\partial h_1}{\partial w_1} = \frac{\partial h_1}{\partial w_1}$$

We can obtain $\frac{\partial h_1}{\partial w_1}$ by looking at the formula of $h_1$:

$$h_1 = \max(0, w_1 x_1 + w_3 x_2 + b_1)$$

Since $w_1 x_1 + w_3 x_2 + b_1 = 1.5 > 0$, the derivative of ReLU is 1 so we get $\frac{\partial h_1}{\partial w_1} = x_1 = 1$. So, $\frac{\partial L}{\partial w_1} = 1$. Now, we can update $w_1$: 

$$w_1 = w_1 - \eta \frac{\partial L}{\partial w_1} = 1 - 0.01 * 1 = 0.99$$

Similarly, we can calculate $\frac{\partial L}{\partial w_3}$:

$$\frac{\partial L}{\partial w_3} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_3} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_5 \cdot \frac{\partial h_1}{\partial w_3} = \frac{\partial h_1}{\partial w_3}$$

But, since $w_1 x_1 + w_3 x_2 + b_1 = 1.5 > 0$, we have $\frac{\partial h_1}{\partial w_3} = x_2 = 0$. So, $\frac{\partial L}{\partial w_3} = 0$. Now, we can update $w_3$:

$$w_3 = w_3 - \eta \frac{\partial L}{\partial w_3} = 1 - 0.01 * 0 = 1$$

*Note that, because the partial derivative was 0, the parameter has not been updated.*

We can now calculate $\frac{\partial L}{\partial b_1}$:

$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial b_1} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_5 \cdot \frac{\partial h_1}{\partial b_1} = \frac{\partial h_1}{\partial b_1}$$

Since $w_1 x_1 + w_3 x_2 + b_1 = 1.5 > 0$, we have $\frac{\partial h_1}{\partial b_1} = 1$. So, $\frac{\partial L}{\partial b_1} = 1$. Now, we can update $b_1$:

$$b_1 = b_1 - \eta \frac{\partial L}{\partial b_1} = 0.5 - 0.01 * 1 = 0.49$$

We are done with the first neuron of the hidden layer, so we can do the exact same thing for the second neuron of the hidden layer, $h_2$:

The loss function at $h_2$ is

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial w_2} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_6 \cdot \frac{\partial h_2}{\partial w_2} = \frac{\partial h_2}{\partial w_2}$$

Since $w_1 x_1 + w_4 x_2 + b_2 = 0.5 > 0$, we have $\frac{\partial h_2}{\partial w_2} = x_1 = 1$. So, $\frac{\partial L}{\partial w_2} = 1$. Now, we can update $w_2$:

$$w_2 = w_2 - \eta \frac{\partial L}{\partial w_2} = 0.5 - 0.01 * 1 = 0.49$$

Similarly, we can calculate $\frac{\partial L}{\partial w_4}$:

$$\frac{\partial L}{\partial w_4} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial w_4} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_6 \cdot \frac{\partial h_2}{\partial w_4} = \frac{\partial h_2}{\partial w_4}$$

Since $w_1 x_1 + w_4 x_2 + b_2 = 0.5 > 0$, we have $\frac{\partial h_2}{\partial w_4} = x_2 = 0$. So, $\frac{\partial L}{\partial w_4} = 0$. Now, we can update $w_4$:

$$w_4 = w_4 - \eta \frac{\partial L}{\partial w_4} = -0.5 - 0.01 * 0 = -0.5$$

Finally, we can calculate $\frac{\partial L}{\partial b_2}$:

$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial b_2} = 2(h_1 w_5 + h_2 w_6 + b_3 - y) w_6 \cdot \frac{\partial h_2}{\partial b_2} = \frac{\partial h_2}{\partial b_2}$$

Since $w_1 x_1 + w_4 x_2 + b_2 = 0.5 > 0$, we have $\frac{\partial h_2}{\partial b_2} = 1$. So, $\frac{\partial L}{\partial b_2} = 1$. Now, we can update $b_2$:

$$b_2 = b_2 - \eta \frac{\partial L}{\partial b_2} = 0 - 0.01 * 1 = -0.01$$

Take a deep breath, we are done...

## Gradient Descent Variants

We have already looked at the basic Gradient Descent algorithm, which updates the parameters after computing the gradients on the entire training dataset. However, this can be computationally expensive for large datasets. Because of this, GD is almost never used when training a neural network. To address this issue, several variants of Gradient Descent have been developed:

* **Stochastic Gradient Descent (SGD):** Instead of computing the gradients on the entire dataset, SGD computes the gradients on a single randomly selected training example (or a randomly selected batch of examples) at each iteration. This allows for much faster updates, but can lead to noisy updates and slower convergence. Surprisingly, this noise can actually help the model escape local minima and find better solutions.
* **Momentum SGD:** This variant of SGD adds a momentum term to the updates, which helps to smooth out the updates and can lead to faster convergence. The formulas are: $v_t = \mu v_{t-1} - \alpha \nabla \mathcal{L}$ and $W = W + v_t$, where $v_t$ is the velocity (momentum) at time $t$, $\mu$ is the momentum coefficient, $\alpha$ is the learning rate, and $\nabla \mathcal{L}$ is the gradient of the loss.

## Data normalization

Before training a neural network and doing backpropagation, it is important to normalize the data. This means that we need to scale the input features to a similar range, usually between 0 and 1 or between -1 and 1. This helps the network to learn faster and can lead to better performance. The way we do this is using Batch Normalization.

Batch Normalization normalizes the activations of a given network layer across the current mini-batch of data. By ensuring that the inputs to a layer have a consistent mean and variance, it mitigates the problem of internal covariate shift. This stabilizes the training process, allows for higher learning rates, and significantly accelerates convergence.

For a mini-batch $B=\{x_1, x_2, \dots, x_m\}$ of size $m$, the transformation is defined by the following steps:

1.  Compute the mini-batch mean:

    $$\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i$$

2.  Compute the mini-batch variance:

    $$\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$$

3.  Normalize the input:

    $$\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$

4.  Scale and shift:

    $$y_i=\gamma\hat{x}_i+\beta$$

Here, $\epsilon$ is a small constant added for numerical stability (preventing division by zero). The variables $\gamma$ (scale) and $\beta$ (shift) are learnable parameters updated during training, ensuring the normalization does not reduce the expressive power of the network.
