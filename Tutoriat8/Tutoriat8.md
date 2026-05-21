# Practical application: Detecting objects in radio signals

In this task you are asked to employ machine learning methods to detect (count) objects in noisy radio signals, where classes represent the number of objects. In this context, a question that arises is whether this task should be treated as a classification or regression task.

To determine whether this task should be treated as a classification or regression problem, we need to consider the nature of the output variable (the number of objects) and the goals of the analysis.

The issue is that we only have 5 classes of objects and we do not know how many objects there are in the signal. If we treat this as a regression problem, we would be trying to predict a continuous value (the number of objects), which may not be appropriate given that the number of objects is discrete and limited to 5 classes. So, this is a *computer vision* classification problem.

For this project, you are to implement 2 different machine learning algoritms to get your points. The first one can be something weaker, like an SVM and you will be good to go. The second one must be the better one. The second model is the one that you will implement and it is going to be pretty hardware-intensive. You will have to use a GPU to train it. In order to make it run on an Nvidia GPU, you need to use the CUDA toolkit, which can be installed into your Python virtual environment with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. If you have an AMD GPU, you can try to use the ROCm toolkit, which is also supported by PyTorch, but I have never tried it, so I cannot give you any advice on how to use it. You can find more information about ROCm on the official PyTorch website.

## Solution: Convolutional Neural Networks (CNNs)

### What are CNNs?
Convolutional Neural Networks (CNNs) are a class of deep neural networks specialized for processing grid-like data, most notably two-dimensional images. They are the foundational architecture for modern computer vision tasks, such as image classification, object detection, and image segmentation. Their primary advantage lies in their ability to automatically and adaptively learn spatial hierarchies of features directly from raw pixel data, eliminating the need for manual feature engineering.

### How do CNNs work?
CNNs process input images through a structured sequence of interconnected layers, progressively extracting increasingly complex visual representations:

1. **Convolutional Layers:** These are the core building blocks. A convolutional layer applies a set of learnable 2D filters (kernels) that slide (convolve) across the width and height of the input image. In early layers, these filters detect low-level features such as edges, lines, and color gradients. In deeper layers, they detect high-level features like shapes or specific object parts. The output is a set of two-dimensional feature maps.

2. **Activation Function:** Following each convolution operation, the output is passed through a non-linear activation function. This introduces non-linearity into the model, enabling the network to learn complex, non-linear relationships in the visual data.

3. **Pooling Layers:** Also known as downsampling layers, these typically follow convolutional layers. They reduce the spatial dimensions (width and height) of the feature maps while retaining the most critical information. For example, Max Pooling extracts the maximum value from a small local window (e.g., 2x2 pixels). This reduces computational complexity, mitigates overfitting, and provides spatial translation invariance, meaning the network can recognize a feature regardless of its exact location in the image.

4. **Fully Connected Layers:** After passing through multiple convolutional and pooling layers, the resulting high-level feature maps are flattened into a single one-dimensional vector. This vector is fed into standard, fully connected neural network layers. The final layer aggregates these features to make the final prediction, often utilizing a Softmax activation function to output class probabilities for image classification tasks.

### Examples of CNN Architectures
- **LeNet-5:** One of the earliest CNN architectures, designed for handwritten digit recognition.
- **AlexNet:** A deeper architecture that won the ImageNet competition in 2012, significantly advancing the field of computer vision.
- **VGGNet:** Known for its simplicity and depth, using small 3x3 convolutional filters.
- **ResNet:** Introduces residual connections to allow for much deeper networks without suffering from vanishing gradients, enabling the training of networks with hundreds of layers.

**!!IMPORTANT!!** For this project, you are not allowed by the rules to use pre-trained or pre-defined CNN architectures. What you can do is implement one of these architectures yourself, but you cannot use any pre-trained weights. You can also design your own architecture, but it must be implemented from scratch without using any pre-trained models.

### Solution structure and description

#### Data Preprocessing

The first thing that we have to do is look at the data and think about how we can make it suitable for our model. We have to look at the data and see if there are any outliers, or if we need to normalize the data. We also have to split the data into training and testing sets.

My reccomendation is to normalize the data, as this will help the model converge faster. We have a bunch of ways in which we can do this:

* **Batch Normalization**
Batch Normalization normalizes the activations of a given network layer across the current mini-batch of data. By ensuring that the inputs to a layer have a consistent mean and variance, it mitigates the problem of internal covariate shift. This stabilizes the training process, allows for higher learning rates, and significantly accelerates convergence.

For a mini-batch $B=\{x_1, x_2, ..., x_m\}$ of size $m$, the transformation is defined by the following steps:

1. Compute the mini-batch mean: 
$$\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i$$


2. Compute the mini-batch variance: 
$$\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$$


3. Normalize the input: 
$$\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$


4. Scale and shift: 
$$y_i=\gamma\hat{x}_i+\beta$$



Here, $\epsilon$ is a small constant added for numerical stability (preventing division by zero). The variables $\gamma$ (scale) and $\beta$ (shift) are learnable parameters updated during training, ensuring the normalization does not reduce the expressive power of the network.

* **Layer Normalization**
Unlike Batch Normalization, Layer Normalization computes the mean and variance across all features (or hidden units) for a *single* training example, completely independent of other examples in the batch. This makes it particularly useful when batch sizes are small, or in architectures where sequence lengths vary dynamically (such as in Transformers, if you want to use them).

For a single input vector $x$ containing $H$ hidden units, the process is:

1. Compute the layer mean: 
$$\mu_L=\frac{1}{H}\sum_{j=1}^{H}x_j$$


2. Compute the layer variance: 
$$\sigma_L^2=\frac{1}{H}\sum_{j=1}^{H}(x_j-\mu_L)^2$$


3. Normalize the input: 
$$\hat{x}_j=\frac{x_j-\mu_L}{\sqrt{\sigma_L^2+\epsilon}}$$


4. Scale and shift: 
$$y_j=\gamma\hat{x}_j+\beta$$



As with Batch Normalization, $\epsilon$ provides numerical stability, while $\gamma$ and $\beta$ are learnable parameters that allow the model to scale and shift the normalized output.

You need to experiment with both of them and see which one works better for your model. You can also try other normalization techniques, such as Instance Normalization or Group Normalization, to see if they improve the performance of your model.

Alright, so we have decided to normalize the data. What else can we do? In order to decide, we need to look at our data. These are radio signals, so we can do horizontal flips, vertical flips, and rotations. We can also add some noise to the data, as this will help the model generalize better. We can also try to do some data augmentation, such as random cropping or random erasing. All of these need to be done with a lower probability, as we do not want to destroy our data. You can find a lot of data augmentation techniques [here](https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html).

We can also resize our images, but we need to be very careful. Our images have very little important information and a lot of noise in them, so we need to think twice about resizing.

Another very important thing that we need to do is to split the training data into 2 different sets: a training set and a validation set. The training set will be used to train the model, while the validation set will be used to evaluate the model during training. This will help us to prevent overfitting and to select the best model. After finishing the training, we can also fine tune the model on the validation data, so that we can get the best possible outcome when generating a solution for the test data.

#### Model Architecture

We are not going to implement convolutional layers by hand.

We are going to use the `torch.nn.Conv2d` module, which is a convolutional layer that applies a 2D convolution over an input signal composed of several input planes.

We are also going to use the `torch.nn.MaxPool2d` module, which is a max pooling layer that applies a 2D max pooling operation over an input signal composed of several input planes. We use pooling to make the training faster, while retaining the information located in that area of the image. 

We are also going to use the `torch.nn.Linear` module, which is a linear layer that applies a linear transformation to the incoming data. This linear layer is a classic neural network layer that actually does the classification. 

We also need to use activation functions to introduce non-linearity into our model. If we do not do this, the entire model is completely equivalent to a single linear layer, which is not what we want. There are a lot of activation functions that we can use:

- [**ReLU**](https://www.youtube.com/watch?v=92pSi7rZJ7c): $f(x) = \max(0,x)$ `torch.nn.ReLU`
- **Leaky [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c)**: $f(x) = \max(0.01x,x)$ `torch.nn.LeakyReLU`
- **ELU** (`torch.nn.ELU`):

$$f(x) = \begin{cases} x & \text{if } x > 0,  \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$
- **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$ `torch.nn.Sigmoid`
- **Tanh**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ `torch.nn.Tanh`
- **GeLU**: $f(x) = x \cdot \Phi(x)$, where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. `torch.nn.GELU`
- **SiLU**: $f(x) = x \cdot \sigma(x)$, where $\sigma(x)$ is the sigmoid function. `torch.nn.SiLU`

You need to experiment with different activation functions and see which one works better for your model. You can also try to use different activation functions in different layers of the model, as this can sometimes improve the performance of the model.

Another thing that we need to do is to use dropout, which is a regularization technique that randomly sets a fraction of the input units to 0 at each update during training time, which helps to prevent overfitting. We can use the `torch.nn.Dropout` module for this.

**Example CNN class**:

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        
        identity = x
        
        out = F.relu(self.conv2(x))
        out = self.conv3(out)
        
        # this is a residual connection
        out += identity
        out = F.relu(out)
        
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

```

This is just an example of a CNN architecture. You can experiment with different architectures and see which one works better for your model. You can also try to use different numbers of layers, different numbers of filters, and different kernel sizes to see how they affect the performance of the model. Something important is that you need the number of filters to increase from the first layer to the next.

Another very funny thing that you can do (and it worked very well for me) is to use randomly generated model architectures. You can generate random architectures by randomly selecting the number of layers, the number of filters, the kernel sizes, and the activation functions. This can be a very good way to find a good architecture for your model, as it allows you to explore a large space of possible architectures. If you want to do this, you need to implement a mechanism called *early stopping*, which will stop the training process if the model does not improve for a certain number of epochs. This will help you to save time and computational resources, as you will not have to train models that are not improving.

#### Hyperparameters

We need to define some more things before actually training our model.

1. *Batch size*: This is the number of samples that will be propagated through the network at once. A common choice is 32 or 64, but you can experiment with different batch sizes to see which one works better for your model. **BIG DISCLAIMER:** If you don't have a GPU (or a powerful one), you should use a smaller batch size, such as 16 or 8, to avoid running out of memory (the famous OOM error).
2. *Number of epochs*: This is the number of times that the entire training dataset will be passed through the network. You need to experiment with different numbers of epochs to see which one works better for your model. You can also use early stopping to stop the training process if the model does not improve for a certain number of epochs.
3. *Learning rate*: This is the step size at which the model's parameters are updated during training. Common choises are 0.01, 0.001, 0.0001, 0.0003, but you can experiment with different learning rates to see which one works better for your model. You can also use learning rate schedulers to adjust the learning rate during training, which can help to improve the performance of the model. I would highly recommend you use `torch.optim.lr_scheduler.CosineAnnealingLR`, which is a learning rate scheduler that adjusts the learning rate according to a cosine annealing schedule. This means that the learning rate will decrease following a cosine curve, which can help to improve the performance of the model by allowing it to converge more smoothly. A similar one is `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`, which is a learning rate scheduler that adjusts the learning rate according to a cosine annealing schedule with warm restarts. This means that the learning rate will decrease following a cosine curve, but it will also periodically reset to the initial learning rate, which can help to improve the performance of the model by allowing it to escape local minima and converge more effectively. Another thing that can work is a procedure called *warmup*, which is a technique that gradually increases the learning rate from a small value to the initial learning rate over a certain number of iterations or epochs at the beginning of training. This can help to improve the performance of the model by allowing it to start with a smaller learning rate, which can help to stabilize the training process and prevent divergence, especially when using large batch sizes or complex architectures.
4. *Optimizer*: This is the algorithm that will be used to update the model's parameters during training. The best optimizers for computer vision tasks are momentum SGD and AdamW. Momentum SGD is a variant of stochastic gradient descent that incorporates momentum, which helps to accelerate convergence and navigate through local minima. AdamW is an optimization algorithm that combines the benefits of Adam with weight decay regularization, which can help to improve generalization performance. You can experiment with different optimizers to see which one works better for your model. If you want to use momentum SGD, I would advise you to define it like this: `torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, nesterov = True)`. If you want to use AdamW, I would advise you to define it like this: `torch.optim.AdamW(model.parameters(), lr = 3e-4, weight_decay = 1e-3)`. (The 3e-4 lr for Adam and AdamW is a joke, but works well enough :) ) You can also experiment with different values of momentum and weight decay to see which ones work better for your model.
5. *Loss function:* This is the function that will be used to measure the error between the model's predictions and the true labels. For a classification task, the most common loss function is Cross Entropy Loss, which can be defined in PyTorch as `torch.nn.CrossEntropyLoss()`. There is an important parameter in the cross entropy loss function called label_smoothing. This parameter is used to smooth the labels, which can help to improve the performance of the model by preventing it from becoming too confident in its predictions. When label smoothing is applied, the true labels are modified to be a weighted average of the original labels and a uniform distribution over all classes. This can help to regularize the model and improve generalization performance. You can experiment with different values of label_smoothing to see which one works better for your model. A common choice is 0.1, but you can try other values as well. Another interesting loss function that you can use is Focal Loss, which is a loss function that is designed to address the class imbalance problem in classification tasks. It does this by down-weighting the loss assigned to well-classified examples, which can help to improve the performance of the model on imbalanced datasets. The dataset in this specific problem has 3500 images for class 1 and 3000 for the rest of the classes, so it is not very imbalanced, but you can still try to use Focal Loss to see if it improves the performance of your model. **Disclaimer:** Focal Loss is not implemented in PyTorch (the library you will probably use), so you will have to implement it yourself. Watch out for those plagiarism checks!

#### Model Training
After defining all of these things, you are ready to train your model. 

It is highly recommended to use a GPU for training your model, as it will significantly speed up the training process. If you do not have a GPU, you can still train your model on a CPU, but it will take much longer. If you do not have a powerful GPU, you can also try to use a cloud service such as Google Colab or Kaggle Kernels, which provide free access to GPUs for a limited time (so be careful what you do with them, as you can easily run out of your free time if you are not careful).

You can use the following code to train your model:

```python
import torch

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If you don't have a GPU, I think there is some experimental thing for integrated GPUs, but I have never tried it. I think it is called torch.xpu_is_available() or something like that.
    model = model.to(device)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
```

#### Other things you can try

- **Ensembles**: Ensembling is a technique that combines the predictions of multiple models to improve the overall performance. You can train multiple models with different architectures, hyperparameters, or random seeds, and then combine their predictions using techniques such as majority voting, averaging, or weighted averaging. This can help to reduce the variance of the predictions and improve the generalization performance of the model. For example, you can train 30 different randomly generated architectures, take the top 5 best models based on their validation accuracy, and then ensemble their predictions to get a better performance on the test set. (I do not encourage gambling, but, in this case, it is acceptable :) ) **Disclaimer:** Ensembling can be very computationally expensive, as it requires training multiple models, so you need to be careful when using this technique, especially if you do not have access to powerful hardware.
- **MixUp/CutMix**: MixUp and CutMix are data augmentation techniques that create new training samples by combining pairs of images and their corresponding labels. MixUp generates new samples by taking a weighted average of two images and their labels, while CutMix creates new samples by cutting and pasting a patch from one image onto another image, and adjusting the labels accordingly. These techniques can help to improve the generalization performance of the model by providing it with more diverse training examples and encouraging it to learn more robust features. You can experiment with these techniques to see if they improve the performance of your model. Unfortunately, you need to implement these techniques yourself, as they are not implemented in PyTorch. Watch out for those plagiarism checks!
- **Transformers**: Transformers are a type of neural network architecture that has been shown to be very effective for a wide range of tasks, including computer vision. They are based on the self-attention mechanism, which allows them to capture long-range dependencies in the data. You can experiment with using transformers for your model to see if they improve the performance. **Disclaimer:** You have pretty limited data, so transformers require a lot of data, so they might not perform very well.
- **Weight Initialization**: The way you initialize your network's weights before the very first epoch can drastically impact whether your model converges quickly or stalls out completely. If weights are initialized too small, the signal vanishes during backpropagation; if they are too large, the signal explodes. To prevent this, you should pair your initialization strategy with your chosen activation functions:
    - **Xavier (Glorot) Initialization**: This technique is designed to keep the variance of the inputs and gradients consistent across layers. It is the optimal choice when your network uses symmetric, zero-centered activation functions like **Sigmoid** or **Tanh**.
    - **Kaiming (He) Initialization**: Because activation functions like [**ReLU**](https://www.youtube.com/watch?v=92pSi7rZJ7c) and **Leaky [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c)** zero out the negative half of the input, they effectively cut the variance of the signal in half. Xavier initialization does not account for this loss of variance, which can lead to vanishing gradients in deeper networks. Kaiming initialization specifically scales the weights to restore that lost variance.

### How can we fix the model if it is not performing well?

Let's address the elephant in the room: what if you have trained your model, but you have realized it has like 50% accuracy. This is a very common situation, especially if you are new to machine learning and computer vision. Do not worry, it is completely normal to have a model that does not perform well at the beginning. The important thing is to analyze the model and try to understand why it is not performing well, and then try to fix it.

The first thing that you should do is to look at the training and validation loss curves. If the training loss is decreasing, but the validation loss is increasing or stagnating, then it is a sign that your model is overfitting (which means that the model learned the training data by heart, but cannot generalize to new data). In this case, you can try to use more regularization techniques, such as bigger dropout, bigger weight decay, or more data augmentation to prevent overfitting. You can also try to use a simpler model architecture, as a more complex model is more likely to overfit, but be careful, as a model that is too simple might not be able to learn the data well enough, which is called underfitting.

But still, usually we want our model to learn the train data well enough anyway. Here we might enounter some problems:

* **Training loss increases**. This usually indicates that the learning rate is too high. To fix this, decrease the learning rate.
* **Training loss decreases slowly**. This usually means that the learning rate is too low. To fix this, increase the learning rate.
* **Training loss decreases then oscillating plateau**. This usually indicates that the learning rate is too high. To fix this, decrease the learning rate to a half or a third of the original. If you are using a learning rate scheduler with warmup, try to lower the maximum learning rate value.
* **Training loss decreases then smooth plateau**. This may indicate a learning rate that is too low. To fix this, increase the learning rate. If you are using a learning rate scheduler with warmup, try to increase the peak learning rate value.
* **Training loss still plateaus after increasing LR**. This means that the capacity of the model is too small, so you need to add more layers or make the layers larger. It can also mean that you are doing too much regularization, so you can make the weight decay smaller or the dropout smaller.

Let's hope that you have finally fixed the training loss. Now, the training loss decreases normally, but there might be some problems with the validation loss:
* **Validation loss is similar to the training loss**. This is a very good sign, so you have done a good job. But as this is your first ever ML project, this probably will not happen first try :)
* **Validation loss is much higher than the training loss**. This usually indicates that the model is overfitting. To fix this, you can try to use more regularization techniques, such as bigger dropout, bigger weight decay, or more data augmentation. You can also try to use a simpler model architecture, as a more complex model is more likely to overfit, but be careful, as a model that is too simple might not be able to learn the data well enough.
* **Validation loss tracks train loss then diverges**. This usually means that the model is starting to overfit, so we stop the training or we implement the other overfitting fixes that we have talked about.
* **Both validation loss and accuracy increase**. This is ok, we can use a model that has a higher validation accuracy as long as the validation loss does not degrade significally.
* **Validation loss is lower than the training loss**. This is fine, because this effect can happen due to heavy data augmentation and regularization, which are disabled during validation.
* **Validation loss is way lower than the training loss**. This means that the validation dataset is way easier for the model than the training set, so you should check if there is some data leakage between the training and validation sets, or if the validation set is not representative of the training set. You can also try to use a different validation set to see if the problem persists.

If nothing works, we can also look at the gradients inside the model to see if the model's weights are being updated properly or not. There could be 2 main problems here:
* **Exploding Gradients**: This can be identified with the loss/metric curves as well. To fix this, we can reduce the learning rate or we can use a technique called gradient clipping, which is a technique that clips the gradients during backpropagation (we will talk about this next time, when we will learn about neural networks) to prevent them from becoming too large. This can help to stabilize the training process and prevent divergence. In PyTorch, you can use `torch.nn.utils.clip_grad_norm_` to clip the gradients.
* **Vanishing Gradients**: This means that some gradients are becoming too small, which can prevent the model from learning effectively. This can be caused by the model being too deep or some activation functions which kill gradients, such as sigmoid, tanh, or [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c) (because of the dying [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c) problem caused by the 0 gradient for negative inputs). To fix this, we can try to use different activation functions, such as Leaky [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c), ELU, GeLU, or SiLU, which do not have the dying [ReLU](https://www.youtube.com/watch?v=92pSi7rZJ7c) problem. We can also try to use a different model architecture that is not too deep, or we can try to use residual connections, which can help to mitigate the vanishing gradient problem by allowing gradients to flow more easily through the network.

Another common problem is that your model can be biased towards one of the classes. If you suspect that this may be a problem, you can print out the precision and the recall for every class after every training epoch. If you see any biasing going on, you can penalize the model for guessing the biased class using some predefined weights.

**!Caution note!** If you see that your laptop/PC is very hot, you can find the temperature of your Nvidia GPU with `nvidia-smi` in the terminal. If the temperature is above 90 degrees Celsius, you should stop the training and let your laptop cool down, as it can damage your hardware. You can also try to use a smaller batch size or a smaller model architecture to reduce the computational load on your GPU.

*Quick note*: For your project, it is very easy to get around 65% - 70% accuracy without much effort. I literally trained a vibe-coded model (with a lot of explainations from me) for 80 epochs and got 68% accuracy and it was nowhere near close to overfitting. I stopped training because I needed my laptop for other stuff and it is not really usable while training a ML model. Today's LLMs are not that good at giving you solutions for computer vision problems, because they do not really understand the data, but they can help you write the code.

For the documentation part, I am giving you a link to [my project from last year](https://github.com/TeodorLepadatu/Deepfake_Image_classifier) so that you have a model. The task was different, but it was still computer vision and they still want the same things written in the documentation.

Good luck with your project, and do not hesitate to ask for help if you need it!
