import pickle
from utils import tensor_from_sentence_pos
from classifier import train_probe


# train with all vs. half of the training data


for i in range(0,13):

    print("Layer: {0} -- N/H, E/H".format(i))

    with open('pickle/train{0}.sentences'.format(i), 'rb') as train_file:
        sentences = pickle.load(train_file)

    with open('pickle/dev{0}.sentences'.format(i), 'rb') as train_file:
        sentences_val = pickle.load(train_file)

    X_normal, y_normal = tensor_from_sentence_pos(sentences)
    X_val_normal, y_val_normal = tensor_from_sentence_pos(sentences_val)

    size = int(list(X_normal.shape)[0] / 2)
    X_normal_reduced = X_normal[0:size]
    y_normal_reduced = y_normal[0:size]

    train_probe(X_normal, y_normal, X_val_normal, y_val_normal)
    train_probe(X_normal_reduced, y_normal_reduced, X_val_normal, y_val_normal)