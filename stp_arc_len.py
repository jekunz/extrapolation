import torch
from utils import tensor_from_sentence_stp, get_labels


# get m_1 and m_2 for the automated splitting points
def get_stats(sentences):
    total_ls = []
    for s in sentences:
        len_deps = [abs(abs(d[0]) - abs(d[1])) for d in s.dependencies]
        total_ls.extend(len_deps)
    std = sorted(total_ls)
    m1 = std[int(len(std)/2)]
    m2 = std[int(len(std)*0.75)]
    return m1, m2


# get X/y for short arcs
def label_data_short(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    m1, _ = get_stats(sentences)

    for s in sentences:
        short_deps = [d for d in s.dependencies if abs(abs(d[0]) - abs(d[1])) < m1 and d[1] != 0]
        if len(short_deps) > 1:
            pairs, labels = get_labels(short_deps, s)
            X_train = torch.cat((X_train, pairs), 0)
            y_train = torch.cat((y_train, labels), 0)

    return X_train, y_train


# get X/y for long arcs
def label_data_long(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    _, m2 = get_stats(sentences)

    for s in sentences:
        long_deps = [d for d in s.dependencies if abs(abs(d[0])-abs(d[1])) > m2 and d[1] != 0]
        if len(long_deps) > 1:
            pairs, labels = get_labels(long_deps, s)
            X_train = torch.cat((X_train, pairs), 0)
            y_train = torch.cat((y_train, labels), 0)

    return X_train, y_train


def make_sets(sentences, sentences_dev):

    X,y = tensor_from_sentence_stp(sentences)
    X_dev, y_dev = tensor_from_sentence_stp(sentences_dev)
    X_short, y_short = label_data_short(sentences)
    X_long, y_long = label_data_long(sentences_dev)

    return X, y, X_short, y_short, X_dev, y_dev, X_long, y_long

