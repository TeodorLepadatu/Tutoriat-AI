import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix


def solve():
    # citim datele din fisiere
    train_df_full = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # impartim setul de date in 80% antrenare si 20% validare
    train_df, val_df = train_test_split(train_df_full, test_size=0.2, random_state=42)

    # extragem caracteristicile folosind bag of words
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(train_df['sample'])
    x_val = vectorizer.transform(val_df['sample'])

    # evaluare pentru dialect
    y_train_dialect = train_df['dialect']
    y_val_dialect = val_df['dialect']

    nb_dialect = MultinomialNB()
    nb_dialect.fit(x_train, y_train_dialect)
    preds_val_dialect = nb_dialect.predict(x_val)

    print("--- evaluare dialect ---")
    print("accuracy:", accuracy_score(y_val_dialect, preds_val_dialect))
    print("precision (macro):", precision_score(y_val_dialect, preds_val_dialect, average='macro'))
    print("recall (macro):", recall_score(y_val_dialect, preds_val_dialect, average='macro'))
    print("f1 (binary):", f1_score(y_val_dialect, preds_val_dialect, average='binary'))
    print("matrice de confuzie dialect:\n", confusion_matrix(y_val_dialect, preds_val_dialect))
    print("\nraport complet dialect:\n", classification_report(y_val_dialect, preds_val_dialect))

    # evaluare pentru categorie
    y_train_category = train_df['category']
    y_val_category = val_df['category']

    nb_category = MultinomialNB()
    nb_category.fit(x_train, y_train_category)
    preds_val_category = nb_category.predict(x_val)

    print("\n--- evaluare categorie ---")
    print("accuracy:", accuracy_score(y_val_category, preds_val_category))
    print("precision (weighted):",
          precision_score(y_val_category, preds_val_category, average='weighted', zero_division=0))
    print("recall (weighted):", recall_score(y_val_category, preds_val_category, average='weighted', zero_division=0))
    print("f1 (weighted):", f1_score(y_val_category, preds_val_category, average='weighted', zero_division=0))
    print("matrice de confuzie categorie:\n", confusion_matrix(y_val_category, preds_val_category))
    print("\nraport complet categorie:\n", classification_report(y_val_category, preds_val_category, zero_division=0))

    # reantrenam modelele pe intregul set de date train pentru performanta maxima pe test
    # observatie: putem face asta pentru ca naive bayes invata repede, iar setul de date este mic
    x_train_full = vectorizer.fit_transform(train_df_full['sample'])
    x_test_final = vectorizer.transform(test_df['sample'])

    nb_dialect.fit(x_train_full, train_df_full['dialect'])
    preds_dialect_final = nb_dialect.predict(x_test_final)

    nb_category.fit(x_train_full, train_df_full['category'])
    preds_category_final = nb_category.predict(x_test_final)

    # construim rezultatul final
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
    print("\npredictiile pentru submission au fost generate cu succes.")


if __name__ == '__main__':
    solve()