# Naive Bayes

**Naive Bayes** is a probabilistic classifier based on **Bayes' Theorem**. Despite its simplicity, it performs surprisingly well on many real-world tasks like spam filtering, sentiment analysis, and document classification.

The core idea: *given some observed features, what is the most probable class?*

---

## The Math Behind It

Bayes' Theorem gives us:

$$P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}$$

| Term | Name | Meaning |
|---|---|---|
| $P(C \mid X)$ | **Posterior** | Probability of class $C$ given features $X$ |
| $P(X \mid C)$ | **Likelihood** | Probability of seeing features $X$ in class $C$ |
| $P(C)$ | **Prior** | Overall probability of class $C$ |
| $P(X)$ | **Evidence** | Probability of features $X$ (same for all classes, so we ignore it) |

Since $P(X)$ is constant across classes, we simplify to:

$$P(C \mid X) \propto P(C) \cdot P(X \mid C)$$

We pick the class with the **highest posterior**:

$$\hat{C} = \underset{C}{\arg\max} \; P(C) \cdot P(X \mid C)$$

---

## The "Naive" Assumption

The "naive" part: we assume all features are **conditionally independent** given the class. That means:

$$P(X \mid C) = P(x_1 \mid C) \cdot P(x_2 \mid C) \cdots P(x_n \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$

This is almost never true in reality, but it makes the math tractable and works well in practice.

---

## Variants of Naive Bayes

| Variant | When to Use | How $P(x_i \mid C)$ is modeled |
|---|---|---|
| **Multinomial NB** | Text classification (word counts) | Multinomial distribution |
| **Bernoulli NB** | Binary features (word present/absent) | Bernoulli distribution |
| **Gaussian NB** | Continuous features | Gaussian (normal) distribution |
| **Complement NB** | Imbalanced text datasets | Complement class statistics |

## Summary

```
Training:
  1. Compute P(C) for each class           ← prior
  2. Compute P(word | C) for each word     ← likelihood (with smoothing)

Prediction for new document X = {x₁, x₂, ..., xₙ}:
  For each class C:
    score(C) = log P(C) + Σ log P(xᵢ | C)
  Return class with highest score
```

## How do we use the Naive Bayes model?

### Problem

We have a dataset with sentences in Romanian and Moldovan. The words in the sentences are shuffled and encrypted. 

- Task 1: We want to classify these sentences by language (1 → Romanian, 2 → Moldovan).
- Task 2: We want to classify these sentences by topic (1 → culture, 2 → finance, 3 → politics, 4 → science, 5 → sports, 6 → technology).

### Solution

The model cannot learn anything from the raw words. Therefore, we need to do some preprocessing to turn the data from text into numbers, as the model will not be able to understand anything from the raw text. 

**Remark 1:** As we know, Moldovan is a dialect of Romanian, so most words are similar, but some other words are very different.

**Remark 2:** The topics required for task 2 are very different, so there would be very different words appearing in each sample.

We are going to use a preprocessing algorithm called *Bag of words* (BoW). It works by converting the text into a collection of words and counts how often each word appears in the text. It ignores word order and grammar, focusing only on frequency. Because the words are shuffled, no context can be retrieved from the data. We can also use an algorithm called *TF-IDF* for this problem, but this algorithm will not appear in your exams (unlike *Bag of words* which has a high chance of appearing).

While Bag of Words only counts frequencies, *TF-IDF* also evaluates how informative a word is across the entire dataset. The metric is composed of two parts:

* **Term Frequency (TF):** Measures how frequently a term occurs in a specific document.
  $$TF(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total number of words in document } d}$$

* **Inverse Document Frequency (IDF):** Measures how important a term is across the entire corpus. It penalizes highly frequent words (e.g., "the", "is") and scales up rare ones that might contain more domain-specific information.
  $$IDF(t) = \log\left(\frac{N}{DF(t)}\right)$$
  Where $N$ is the total number of documents and $DF(t)$ is the number of documents containing the term $t$.

The final feature representation for a word is the product of these two metrics:

$$TF\text{-}IDF(t, d) = TF(t, d) \cdot IDF(t)$$

Now let's return to our task.
After preprocessing using *Bag of words*, we give the numbers to the naive bayes model and we can generate the predictions.

### Evaluating the solution

After the model has outputted its predictions, we need to have a way of figuring out if the predictions were good or not. We use the following metrics to measure classification performance:

* **True Positives ($TP$):** The model correctly predicted the positive class (e.g., predicting an email is spam, and it actually is spam).
* **True Negatives ($TN$):** The model correctly predicted the negative class (e.g., predicting an email is not spam, and it actually is not).
* **False Positives ($FP$):** The model incorrectly predicted the positive class (e.g., predicting an email is spam, but it is actually a regular email). 
* **False Negatives ($FN$):** The model incorrectly predicted the negative class (e.g., predicting an email is not spam, but it actually is spam). 

* **Accuracy:** The ratio of correctly predicted data points to the total number of data points. It provides a general overview of model performance but can be misleading if the dataset is highly imbalanced.

  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

* **Precision:** The ratio of correctly predicted positive observations to the total predicted positive observations. It measures the accuracy of the positive predictions.

  $$\text{Precision} = \frac{TP}{TP + FP}$$

* **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all actual observations in that class. It measures the model's ability to identify all relevant instances of a class.

  $$\text{Recall} = \frac{TP}{TP + FN}$$

* **F1 Score:** The harmonic mean of Precision and Recall. It provides a single metric that balances both precision and recall, making it highly effective for evaluating models on imbalanced datasets.

  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Depending on whether we are dealing with a binary classification (Task 1) or multi-class classification (Task 2), we use specific averaging methods to calculate these metrics across classes. Let $C$ represent the total number of classes, $S_i$ represent the support (number of true instances) for class $i$, and $N$ represent the total number of instances. These averaging methods can be applied to Precision, Recall, and F1 Score:

* **Binary:** Applies only to binary classification tasks. It calculates the metric strictly for the class designated as the "positive" target class, ignoring the performance on the negative class. The formulas are identical to the base formulas provided above.

* **Macro-Averaging:** Calculates the target metric independently for each class and then takes the unweighted average. It treats all classes equally, regardless of their support. 

  $$\text{Macro-Precision} = \frac{1}{C} \sum_{i=1}^{C} \text{Precision}_i$$

  $$\text{Macro-Recall} = \frac{1}{C} \sum_{i=1}^{C} \text{Recall}_i$$

  $$\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^{C} \text{F1}_i$$

* **Micro-Averaging:** Aggregates the contributions of all classes (summing all $TP$, $FP$, and $FN$) to compute the average metric. Note that in micro-averaging, the overall precision, recall, and F1 score will all equal the overall accuracy.

  $$\text{Micro-Precision} = \frac{\sum_{i=1}^{C} TP_i}{\sum_{i=1}^{C} TP_i + \sum_{i=1}^{C} FP_i}$$

  $$\text{Micro-Recall} = \frac{\sum_{i=1}^{C} TP_i}{\sum_{i=1}^{C} TP_i + \sum_{i=1}^{C} FN_i}$$

  $$\text{Micro-F1} = 2 \cdot \frac{\text{Micro-Precision} \cdot \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}$$

* **Weighted-Averaging:** Calculates the metric independently for each class but averages them using weights proportional to the support ($S_i$) of each class. This is highly useful for imbalanced datasets as it prevents classes with very few instances from heavily skewing the results.

  $$\text{Weighted-Precision} = \sum_{i=1}^{C} \frac{S_i}{N} \text{Precision}_i$$

  $$\text{Weighted-Recall} = \sum_{i=1}^{C} \frac{S_i}{N} \text{Recall}_i$$

  $$\text{Weighted-F1} = \sum_{i=1}^{C} \frac{S_i}{N} \text{F1}_i$$

* **Confusion Matrix:** A table used to visualize the performance of a classification model. It displays the true positives, true negatives, false positives, and false negatives, allowing you to see exactly where the model is confusing specific classes with one another.

* **Classification Report:** A summary provided by scikit-learn that displays the precision, recall, F1-score, and support (the number of true instances) for each individual class in the dataset.

To understand how these metrics apply to multi-class classification (where you have more than two categories, such as Task 2 with 6 topics), you need to use a **One-vs-Rest** approach. 

In a binary problem, you have a clear "Positive" and "Negative" class. In a multi-class problem, there is no single positive class. Instead, you calculate the metrics for *each class individually* by treating that specific class as the "Positive" class and grouping all other classes together as the "Negative" class.



For a given class, let's call it **Class A**, the components are defined as follows:

* **True Positives ($TP_A$):** The model correctly predicted Class A. (Actual = A, Predicted = A)
* **True Negatives ($TN_A$):** The model correctly predicted that the instance was *not* Class A. (Actual $\neq$ A, Predicted $\neq$ A)
* **False Positives ($FP_A$):** The model incorrectly predicted Class A. (Actual $\neq$ A, Predicted = A)
* **False Negatives ($FN_A$):** The model failed to predict Class A when it should have. (Actual = A, Predicted $\neq$ A)

1.  **Per-Class Calculation:** First, using the definitions above, the model calculates the basic metrics (Precision, Recall, F1-Score) strictly for Class A. 
    * $\text{Precision}_A = \frac{TP_A}{TP_A + FP_A}$
2.  **Repeat for All Classes:** The model repeats this process for Class B, Class C, and so on, until every class has its own Precision, Recall, and F1-Score.
3.  **Aggregation:** Because having a list of metrics for every single class is difficult to evaluate as a single model score, you aggregate these individual scores using the averaging methods you provided:
    * **Macro-Average:** You take the unweighted mean of all the individual class scores. This gives equal importance to every class, making it useful if you care about minority classes just as much as majority classes.
    * **Weighted-Average:** You take the mean of all the individual class scores, but multiply each by the proportion of instances that actually belong to that class. This favors the majority classes.
    * **Micro-Average:** Instead of averaging the final percentages, you sum up all the global $TP$, $FP$, and $FN$ across all classes first, and then apply the standard formula. In multi-class problems where every instance is assigned exactly one label, Micro-Precision, Micro-Recall, and Micro-F1 will mathematically reduce to the overall **Accuracy** of the model.

A **Confusion Matrix** is the standard way to visualize this multi-class behavior. In a multi-class confusion matrix, the $TP$ for a specific class is located on the diagonal. The $FPs$ for that class are the sum of the other values in its column, and the $FNs$ are the sum of the other values in its row.

We do not know the labels for the test set, so we are going to split the training set into a train set and a validation set. Firstly, we will only train the model on the train set and evaluate it on the validation set using these metrics. Afterwards, to get the best possible performance on the test set, we will train on the unsplit (original) train set.

#### Results 
If you implement the bag of words + naive bayes solution, you will get about 75% for all metrics in the language classification and 66% accuracy, 46% precision, 49% recall and 47% f1 for the topic classification. The implementation is in the *language_topic* folder. If you implement TF-IDF + Naive Bayes, you will get better results overall.

### Exercise for you

We have given you an exercise to try to test yourselves on this type of tasks. In the *Exercise* folder you have a csv file with a dataset taken from Reddit. Your job is to classify these posts and figure out if the author is anxious or not. You should get more than 75% accuracy if you implement the same approach as in the problem presented here. You might get better results using TFIDF, but we have not tried this approach so no guarantees here.

