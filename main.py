import pickle
from classifier import train_probe
from scoring_functions.pos_sen_len_ling import make_sets


# iterate over all layers and train/evaluate probes
for i in range(0,13):

    print("Layer: {0} -- Standard, Control, Extrapolation".format(i))

    with open('pickle2/train{0}.sentences'.format(i), 'rb') as train_file:
        sentences = pickle.load(train_file)

    with open('pickle2/dev{0}.sentences'.format(i), 'rb') as train_file:
        sentences_val = pickle.load(train_file)

    X_normal, y_normal, X_easy, y_easy, X_val_normal, y_val_normal, X_val_hard, y_val_hard = make_sets(sentences,
                                                                                                       sentences_val)

    train_probe(X_normal, y_normal, X_val_normal, y_val_normal)
    train_probe(X_normal, y_normal, X_val_hard, y_val_hard)
    train_probe(X_easy, y_easy, X_val_hard, y_val_hard)
