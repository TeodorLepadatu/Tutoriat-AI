import numpy as np
from sklearn.svm import SVC

with open("train_sentences.txt", "r") as f:
    train_sentences = [line.strip() for line in f.readlines() if line.strip()]

with open("test_sentences.txt", "r") as f:
    test_sentences = [line.strip() for line in f.readlines() if line.strip()]

with open("words.txt", "r") as f:
    grams = [line.strip() for line in f.readlines() if line.strip()]

train_labels = np.load('train_labels.npy', allow_pickle=True)

with open("mapping.txt", "r") as f:
    mapp = [line.strip().split(",") for line in f.readlines() if line.strip()]
    mapp[9] = [",", '10'] # caracterul e virgula si strica smecheria
    mapp[17] = [" ", '18'] # nu ia spatiu in considerare
    mapping = {}
    for i in range(256):
        mapping[chr(i)] = 0
    for m in mapp:
        mapping[m[0]] = int(m[1])
        mapping[m[0].upper()] = int(m[1])

print(mapping)

def convolution(text, gram):
    lt = len(text) - len(gram) + 1
    lg = len(gram)
    convolution = []

    normgram = 0
    for j in range(lg):
        if gram[j] in mapping.keys():
            normgram += mapping[gram[j]] ** 2
    normgram = normgram ** 0.5

    for i in range(lt):
        conv = 0
        normtext = 0
        for j in range(lg):
            if text[i+j] in mapping.keys():
                normtext += mapping[text[i+j]] ** 2
        normtext = normtext ** 0.5

        for j in range(lg):
            if text[i + j] in mapping.keys() and gram[j] in mapping.keys():
                conv += mapping[text[i+j]] * mapping[gram[j]]
        conv /= (normgram * normtext + 1e-6)
        convolution.append(conv)

    big_values = 0
    for i in convolution:
        if i > 0.9:
            big_values += 1

    return big_values

def vector500(dataset):
    new_dataset = []
    i = 1
    for doc in dataset:
        new_doc = []
        for gram in grams:
            new_doc.append(convolution(doc, gram))
        new_dataset.append(new_doc)
        print(i)
        i += 1
    return np.array(new_dataset)

train500 = vector500(train_sentences)
test500 = vector500(test_sentences)

def hellinger(dataset1, dataset2):
    s1 = dataset1.shape[0]
    s2 = dataset2.shape[0]
    K = np.zeros([s1, s2])
    for i, d1 in enumerate(dataset1):
        for j, d2 in enumerate(dataset2):
            K[i, j] = np.sum(np.sqrt(d1 * d2))
    return K

train_matrix = hellinger(train500, train500)
test_matrix = hellinger(test500, train500)

model = SVC(C=3, kernel="precomputed")
model = model.fit(train_matrix, train_labels)
predictions = model.predict(test_matrix)

with open("Duzi_MihaiNicolae_subiect4_solutia1.txt", "w") as f:
    for label in predictions:
        f.write(f"{label.item()}\n")