import torch
import word_representations

'''

load data; 
for original UD files in conllu format, e.g. 'en_ewt-ud-train.conllu'.
save as Sentence object that contains BERT states, POS tags and dependencies

'''


class Sentence:
    def __init__(self, tokens, pos, states, dependencies):
        self.tokens = tokens
        self.pos = pos
        self.states = states
        self.dependencies = dependencies

    def __str__(self):
        return ' '.join(t for t in self.tokens)

    # states and labels for labels task
    def labels(self):
        dep_pairs = []
        labels = []
        label_ind = []
        for d in self.dependencies:
            dep_pairs.append(torch.cat((self.states[d[0]-1], self.states[d[1]-1]), 0))
            labels.append(d[2])
        for l in labels:
            label_ind.append(label_to_ix[l])

        return torch.stack(dep_pairs), torch.LongTensor(label_ind)

    # states and tags for pos task
    def pos_tagging(self):
        return torch.stack(self.states[0:len(self.pos)]), torch.LongTensor(self.pos)


def pos_to_ix(tag):
    tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']
    tag_to_ix = dict(zip(tags, list(range(len(tags)))))
    tag_index = tag_to_ix[tag]
    return torch.tensor(tag_index, dtype=torch.long)


def label_to_ix(label):
    labels = ['acl', 'acl:relcl', 'advcl', 'advmod', 'advmod:emph', 'advmod:lmod', 'amod', 'appos', 'aux', 'aux:pass',
                'case', 'cc', 'cc:preconj', 'ccomp', 'clf', 'compound', 'compound:lvc', 'compound:prt', 'compound:redup',
                'compound:svc', 'conj', 'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'det:numgov', 'det:nummod', 'det:poss',
                'discourse', 'dislocated', 'expl', 'expl:impers', 'expl:pass', 'expl:pv', 'fixed', 'flat', 'flat:foreign',
                'flat:name', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubj:pass',
                'nummod', 'nummod:gov', 'obj', 'obl', 'obl:agent', 'obl:arg', 'obl:lmod', 'obl:tmod', 'orphan', 'parataxis',
                'punct', 'reparandum', 'root', 'vocative', 'xcomp']
    label_to_ix = dict(zip(labels, list(range(len(labels)))))
    label_index = label_to_ix[label]
    return torch.tensor(label_index, dtype=torch.long)


def read_conllu(layer, path):
    word_repr = word_representations.Bert(layer)

    with open(path, 'r') as file:
        sentences = []
        pos = []
        deps = []
        tokens = []
        for i, line in enumerate(file):
            if line == '\n':
                text = '[CLS]' + ' '.join(tokens) + '[SEP]'
                pos = [pos_to_ix(tag) for tag in pos]
                sentences.append(Sentence(tokens, pos, word_repr.get_bert(text, layer), deps))
                pos = []
                tokens = []
                deps = []
                continue
            if line[0] == '#':
                continue
            line = line.rstrip('\n')
            line = line.split('\t')
            symbols = ['.', ',', '<', '>', ':', ';', '\'', '/', '-', '_', '%', '@', '#', '$', '^', '*', '?', '!', "‘",
                       "’", "'", "+", '=', '|', '\’']
            if len(line[1]) > 1:
                for sym in symbols:
                    line[1] = line[1].replace(sym, '')
            if line[1] == '':
                line[1] = 'unk'
            tokens.append(line[1])
            pos.append(line[3])
            try:
                if int(line[6]) != 0:
                    deps.append((int(line[0]), int(line[6]), label_to_ix(line[7])))
            except ValueError:
                # print("value error ; the following dependency was not appended:", line[0], line[6], line[7])
                # occurs with index of type '5.1'; rare ; can be ignored
                pass

        return sentences
