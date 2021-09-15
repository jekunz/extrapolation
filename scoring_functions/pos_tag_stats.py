from collections import defaultdict, Counter
from utils import get_tensors


# dict for word form and tag distributions for each word
class TagdictTagger:

    def __init__(self):
        self.tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']
        self.tagdict = {}
        self.most_frequent_tag = None

    def train(self, data):
        tagdict = defaultdict(Counter)
        counter = Counter()
        for tagged_sentence in data:
            for word, tag in zip(tagged_sentence.tokens, tagged_sentence.pos):
                tagdict[word][tag] += 1
                counter[tag] += 1
        self.tagdict = defaultdict(defaultdict)
        for word in tagdict:
            total = sum(tagdict[word].values())
            for tag in tagdict[word]:
                self.tagdict[word][tag] = tagdict[word][tag] / total

    def tag_score(self, word, tag):
        return self.tagdict[word][tag] if tag in self.tagdict[word] else 0


def make_sets(sentences, sentences_val):

    Td = TagdictTagger()
    Td.train(sentences)

    scores = []

    for s in sentences_val:
        if len(s.pos) == len(s.states[:-1]):
            for i in range(len(s.tokens)):
                scores.append(Td.tag_score(s.tokens[i], s.pos[i]))

    scores = sorted(scores)
    scores = list(reversed(scores))
    m_1 = scores[int(len(scores) * 0.5)]
    m_2 = scores[int(len(scores) * 0.75)]

    X_easy = []
    y_easy = []
    X_normal = []
    y_normal = []

    for s in sentences:
        if len(s.pos) == len(s.states[:-1]):
            for i in range(len(s.tokens)):
                X_normal.append(s.states[i])
                y_normal.append(s.pos[i])
                if Td.tag_score(s.tokens[i], s.pos[i]) > m_1:
                    X_easy.append(s.states[i])
                    y_easy.append(s.pos[i])

    X_val_hard = []
    y_val_hard = []
    X_val_normal = []
    y_val_normal = []

    for s in sentences_val:
        if len(s.pos) == len(s.states[:-1]):
            for i in range(len(s.tokens)):
                X_val_normal.append(s.states[i])
                y_val_normal.append(s.pos[i])
                if Td.tag_score(s.tokens[i], s.pos[i]) < m_2:
                    X_val_hard.append(s.states[i])
                    y_val_hard.append(s.pos[i])

    t_X_easy, t_y_easy = get_tensors(X_easy, y_easy)
    t_X_normal, t_y_normal = get_tensors(X_normal, y_normal)
    t_X_val_hard, t_y_val_hard = get_tensors(X_val_hard, y_val_hard)
    t_X_val_normal, t_y_val_normal = get_tensors(X_val_normal, y_val_normal)

    return t_X_normal, t_y_normal, t_X_easy, t_y_easy, t_X_val_normal, t_y_val_normal, t_X_val_hard, t_y_val_hard
