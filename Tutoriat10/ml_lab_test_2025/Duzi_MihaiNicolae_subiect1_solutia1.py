import numpy as np
from sklearn.naive_bayes import MultinomialNB

def normalize_data(train_data, test_data, tip=None):
    if tip == "l1":
        train_norm = np.sum(np.abs(train_data), axis=1, keepdims=True)
        test_norm = np.sum(np.abs(test_data), axis=1, keepdims=True)
        return train_data / train_norm, test_data / test_norm
    elif tip == "l2":
        train_norm = np.linalg.norm(train_data, ord=2, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_data, ord=2, axis=1, keepdims=True)
        return train_data / (train_norm+0.00001), test_data / (test_norm+0.00001)
    return train_data, test_data


class BagOfWords:
    def __init__(self):
        self.vocab = {}

    def build_vocabulary(self, data):
        order_words = []
        for msg in data:
            for word in msg:
                if word not in self.vocab.keys():
                    order_words.append(word)
                    self.vocab[word] = len(order_words)-1
        return order_words

    def get_features(self, data):
        matrix = np.zeros((data.shape[0], len(self.vocab)), dtype=np.float32)
        for i in range(data.shape[0]):
            for word in data[i]:
                if word in self.vocab:
                    matrix[i][self.vocab[word]] += 1
        return matrix

with open("train_sentences.txt", "r") as f:
    train_sentences = [line.strip() for line in f.readlines() if line.strip()]

with open("test_sentences.txt", "r") as f:
    test_sentences = [line.strip() for line in f.readlines() if line.strip()]

train_sentences = np.array(train_sentences)
test_sentences = np.array(test_sentences)
train_labels = np.load('train_labels.npy', allow_pickle=True)

BOW = BagOfWords()
list_vocab = BOW.build_vocabulary(train_sentences)
train_vectors = BOW.get_features(train_sentences)
test_vectors = BOW.get_features(test_sentences)
normal_train, normal_test = normalize_data(train_vectors, test_vectors)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(normal_train, train_labels)
predictions = naive_bayes_model.predict(normal_test)

with open("Duzi_MihaiNicolae_subiect1_solutia1.txt", "w") as f:
    for label in predictions:
        f.write(f"{label.item()}\n")