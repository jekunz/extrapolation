from utils import tensor_from_sentence_pos, get_tensors
from classifier import train_probe


# get the standard training sets
def train_data(sentences, sentences_val):
    X_normal, y_normal = tensor_from_sentence_pos(sentences)
    X_val_normal, y_val_normal = tensor_from_sentence_pos(sentences_val)

    return X_normal, y_normal, X_val_normal, y_val_normal


# rank losses for train data
def rank_data(losses, X,y):
    X_easy = []
    y_easy = []
    losses_mean = []

    for k in range(len(losses[0])):
        s = 0
        for i in range(len(losses)):
            s += losses[i][k]
        losses_mean.append(s / len(losses))

    m_1 = sorted(losses_mean)[int(len(losses_mean)*0.5)]

    for i in range(len(losses_mean)):
        if losses_mean[i] < m_1:
            X_easy.append(X[i])
            y_easy.append(y[i])

    return get_tensors(X_easy, y_easy)


# rank losses for dev data
def rank_dev_data(losses, X,y):
    X_hard = []
    y_hard = []
    losses_mean = []

    for k in range(len(losses[0])):
        s = 0
        for i in range(len(losses)):
            s += losses[i][k]
        losses_mean.append(s / len(losses))

    m_2 = sorted(losses_mean)[int(len(losses_mean)*0.75)]

    for i in range(len(losses_mean)):
        if losses_mean[i] > m_2:
            X_hard.append(X[i])
            y_hard.append(y[i])

    return get_tensors(X_hard, y_hard)


def make_sets(sentences, sentences_val):
    X_normal, y_normal, X_val_normal, y_val_normal = train_data(sentences, sentences_val)
    _, losses, losses_dev = train_probe(X_normal, y_normal, X_val_normal, y_val_normal, batch_size=1, get_dev_loss=True)

    X_easy, y_easy = rank_data(losses,X_normal, y_normal)
    X_val_hard, y_val_hard = rank_dev_data(losses_dev,X_val_normal, y_val_normal)

    return X_normal, y_normal, X_easy, y_easy, X_val_normal, y_val_normal, X_val_hard, y_val_hard