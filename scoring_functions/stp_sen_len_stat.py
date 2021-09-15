import torch
from utils import get_labels, tensor_from_sentence_stp


# get m_1 and m_2 for automatic splitting points
def get_stats(sentences):
    total_ls = []
    for s in sentences:
        len_deps = [len(s.tokens)]*len(s.tokens)
        total_ls.extend(len_deps)
    std = sorted(total_ls)
    m1 = std[int(len(std)/2)]
    m2 = std[int(len(std)*0.75)]
    print(m1, m2)
    return m1, m2


# get X/y for short sentences
def label_short(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    m1, _ = get_stats(sentences)
    #m1 = 23

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

    _, m2 = get_stats(sentences)
    #m2 = 34

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