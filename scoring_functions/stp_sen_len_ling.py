import torch
from utils import get_labels, tensor_from_sentence_stp


# get X/y for short sentences
def label_short(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    m1 = 17

    for s in sentences:
        if len(s.tokens) < m1:
            if len(s.dependencies) > 1:
                pairs, labels = get_labels(s.dependencies, s)
                X_train = torch.cat((X_train, pairs), 0)
                y_train = torch.cat((y_train, labels), 0)

    return X_train, y_train


# get X/y for long sentences
def label_long(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    m2 = 29

    for s in sentences:
        if len(s.tokens) > m2:
            if len(s.dependencies) > 1:
                pairs, labels = get_labels(s.dependencies, s)
                X_train = torch.cat((X_train, pairs), 0)
                y_train = torch.cat((y_train, labels), 0)

    return X_train, y_train


def make_sets(sentences, sentences_dev):

    X,y = tensor_from_sentence_stp(sentences)
    X_dev, y_dev = tensor_from_sentence_stp(sentences_dev)
    X_short, y_short = label_short(sentences)
    X_long, y_long = label_long(sentences_dev)

    return X, y, X_short, y_short, X_dev, y_dev, X_long, y_long