import torch


# convert lists of states/labels to single tensors
def get_tensors(X, y):
    with torch.no_grad():
        X_new = []
        y_new = []
        tensX = torch.Tensor(len(X), 768)
        tensy = torch.LongTensor(len(y), 1)

        for item in X:
            X_new.append(item.unsqueeze(dim=0))
        for item in y:
            y_new.append(item.unsqueeze(dim=0))

        torch.cat(X_new, out=tensX)
        torch.cat(y_new, out=tensy)
        return tensX, tensy


# get X/y from sentence objects (POS)
def tensor_from_sentence_pos(sentences):
    X = []
    y = []

    for s in sentences:
        if len(s.pos) == len(s.states[:-1]):
            X.extend(s.states[:-1])
            y.extend(s.pos)

    return get_tensors(X,y)


# get dep. pairs and labels for a sentence object as tensors
def get_labels(dependencies, sentence):
    dep_pairs = []
    labels = []
    for d in dependencies:
        dep_pairs.append(torch.cat((sentence.states[d[0]-1], sentence.states[d[1]-1]), 0))
        labels.append(d[2])
    return torch.stack(dep_pairs), torch.LongTensor(labels)


# get X/y from sentence objects (STP)
def tensor_from_sentence_stp(sentences):
    X_train = torch.FloatTensor()
    y_train = torch.LongTensor()

    for s in sentences:
        if len(s.dependencies) > 1:
            pairs, labels = get_labels(s.dependencies, s)
            X_train = torch.cat((X_train, pairs), 0)
            y_train = torch.cat((y_train, labels), 0)

    return X_train, y_train
