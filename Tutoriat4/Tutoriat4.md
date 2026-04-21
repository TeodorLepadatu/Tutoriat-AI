# Naive Bayes 

## How do we use the Naive Bayes model?

### Problem

We have a dataset with sentences in Romanian and Moldovan. The words in the sentences are shuffled and encrypted. 

- Task 1: We want to classify these sentences by language (1 → Romanian, 2 → Moldovan).
- Task 2: We want to classify these sentences by topic (1 → culture, 2 → finance, 3 → politics, 4 → science, 5 → sports, 6 → technology).

### Solution

The model cannot learn anything from the raw words. Therefore, we need to do some preprocessing to turn the data from text into numbers, as the model will not be able to understand anything from the raw text. 

**Remark 1:** As we know, Moldovan is a dialect of Romanian, so most words are similar, but some other words are very different.
**Remark 2:** The topics required for task 2 are very different, so there would be very different words appearing in each sample.

We are going to use a preprocessing algorithm called *Bag of words*. It works by converting the text into a collection of words and counts how often each word appears in the text. It ignores word order and grammar, focusing only on frequency. Because the words are shuffled, no context can be retrieved from the data. We can also use an algorithm called *TF-IDF* for this problem, but this algorithm will not appear in your exams (unlike *Bag of words* which has a high chance of appearing).

After preprocessing using *Bag of words*, we give the numbers to the naive bayes model and we can generate the predictions.

### Evaluating the solution

After the model has outputted its predictions, we need to have a way of figuring out if the predictions were good or not. We use the following metrics to measure classification performance. In the formulas below, $TP$ stands for True Positives, $TN$ for True Negatives, $FP$ for False Positives, and $FN$ for False Negatives.

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

We do not know the labels for the test set, so we are going to split the training set into a train set and a validation set. Firstly, we will only train the model on the train set and evaluate it on the validation set using these metrics. Afterwards, to get the best possible performance on the test set, we will train on the unsplit (original) train set.

#### Results 
If you implement the bag of words + naive bayes solution, you will get about 75% for all metrics in the language classification and 66% accuracy, 46% precision, 49% recall and 47% f1 for the topic classification.
