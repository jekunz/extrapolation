from collections import Counter
from utils import tensor_from_sentence_stp, get_tensors
from classifier import train_probe


def train_data(sentences, sentences_val):
    X_normal, y_normal = tensor_from_sentence_stp(sentences)
    X_val_normal, y_val_normal = tensor_from_sentence_stp(sentences_val)

    return X_normal, y_normal, X_val_normal, y_val_normal


# create dict; key: training example (BERT state); value: number of correct classifications
def get_succ_dict(knowns):
    counters = []
    countdict = {}
    traindict = {}

    for k in range(len(knowns[0])):
        tuples = []
        comps = []
        temp = [knowns[i][k] for i in range(len(knowns))]

        for t in temp:
            comps.extend([(elem[0][0].item(), elem[0][1].item()) for elem in t])
            tuples.extend(elem for elem in t)
        for c, t in zip(comps, tuples):
            traindict[c] = t

        c = Counter(comps)
        n_of_counts = [c[tup] for tup in comps]

        map_key_counts = {}
        for tup in comps:
            map_key_counts[tup] = c[tup]
        countdict[k] = map_key_counts

        n_of_counts_c = Counter(n_of_counts)
        for n in n_of_counts_c.keys():
            n_of_counts_c[n] = int(n_of_counts_c[n] / n)
        counters.append(n_of_counts_c)

    succ = Counter()

    for i in range(len(countdict)):
        for e in countdict[i].keys():
            succ[e] += countdict[i][e]

    return succ, traindict


# create easy and hard sets with saved classification results of a probe
def train_rank(evaluators):
    knowns = []
    train_knowns = []

    for e in evaluators:
        lst = [k for k in e.known_ex]
        lst2 = [k for k in e.train_known_ex]
        knowns.append(lst)
        train_knowns.append(lst2)

    succ_train, dict_train = get_succ_dict(train_knowns)
    m_1 = succ_train.most_common()[int(len(succ_train) * 0.5)][1]

    easy = []
    for e in succ_train.keys():
        if succ_train[e] > m_1:
            easy.append(dict_train[e])

    succ_dev, dict_dev = get_succ_dict(knowns)
    m_2 = succ_dev.most_common()[int(len(succ_dev)*0.25)][1]

    hard = []
    for e in succ_dev.keys():
        if succ_dev[e] < m_2:
            hard.append(dict_dev[e])


    X_easy, y_easy = zip(*easy)
    X_hard, y_hard = zip(*hard)

    tensX, tensy = get_tensors(X_easy,y_easy)

    return tensX, tensy, X_hard, y_hard


def make_sets(sentences, sentences_val):
    X_normal, y_normal, X_val_normal, y_val_normal = train_data(sentences, sentences_val)
    evaluators, _, _ = train_probe(X_normal, y_normal, X_val_normal, y_val_normal)
    X_easy, y_easy, X_val_hard, y_val_hard = train_rank(evaluators)

    return X_normal, y_normal, X_easy, y_easy, X_val_normal, y_val_normal, X_val_hard, y_val_hard