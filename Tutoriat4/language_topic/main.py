import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix


def solve():
    # read data from files
    train_df_full = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # split the dataset into 80% training and 20% validation
    train_df, val_df = train_test_split(train_df_full, test_size=0.2, random_state=42)

    # extract features using bag of words
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(train_df['sample'])
    x_val = vectorizer.transform(val_df['sample'])

    # evaluation for dialect
    y_train_dialect = train_df['dialect']
    y_val_dialect = val_df['dialect']

    nb_dialect = MultinomialNB()
    nb_dialect.fit(x_train, y_train_dialect)
    preds_val_dialect = nb_dialect.predict(x_val)

    print("--- dialect evaluation ---")
    print("accuracy:", accuracy_score(y_val_dialect, preds_val_dialect))
    print("precision (macro):", precision_score(y_val_dialect, preds_val_dialect, average='macro'))
    print("recall (macro):", recall_score(y_val_dialect, preds_val_dialect, average='macro'))
    print("f1 (macro):", f1_score(y_val_dialect, preds_val_dialect, average='macro'))
    print("dialect confusion matrix:\n", confusion_matrix(y_val_dialect, preds_val_dialect))
    print("\ncomplete dialect report:\n", classification_report(y_val_dialect, preds_val_dialect))

    # evaluation for category
    y_train_category = train_df['category']
    y_val_category = val_df['category']

    nb_category = MultinomialNB()
    nb_category.fit(x_train, y_train_category)
    preds_val_category = nb_category.predict(x_val)

    print("\n--- category evaluation ---")
    print("accuracy:", accuracy_score(y_val_category, preds_val_category))
    print("precision (macro):",
          precision_score(y_val_category, preds_val_category, average='macro', zero_division=0))
    print("recall (mcaro):", recall_score(y_val_category, preds_val_category, average='macro', zero_division=0))
    print("f1 (macro):", f1_score(y_val_category, preds_val_category, average='macro', zero_division=0))
    print("category confusion matrix:\n", confusion_matrix(y_val_category, preds_val_category))
    print("\ncomplete category report:\n", classification_report(y_val_category, preds_val_category, zero_division=0))

    # retrain the models on the entire train dataset for maximum test performance
    # note: we can do this because naive bayes learns fast, and the dataset is small
    x_train_full = vectorizer.fit_transform(train_df_full['sample'])
    x_test_final = vectorizer.transform(test_df['sample'])

    nb_dialect.fit(x_train_full, train_df_full['dialect'])
    preds_dialect_final = nb_dialect.predict(x_test_final)

    nb_category.fit(x_train_full, train_df_full['category'])
    preds_category_final = nb_category.predict(x_test_final)

    # build the final result
    results = []

    for datapoint_id, pred in zip(test_df['datapointID'], preds_dialect_final):
        results.append({
            'subtaskID': 1,
            'datapointID': datapoint_id,
            'answer': pred
        })

    for datapoint_id, pred in zip(test_df['datapointID'], preds_category_final):
        results.append({
            'subtaskID': 2,
            'datapointID': datapoint_id,
            'answer': pred
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv('submission.csv', index=False)
    print("\npredictions for submission generated successfully.")


if __name__ == '__main__':
    solve()
