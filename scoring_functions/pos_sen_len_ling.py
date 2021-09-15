from utils import tensor_from_sentence_pos


def make_sets(sentences, sentences_val):

    short_sentences = []
    long_sentences = []

    for s in sentences:
        if len(s.tokens) < 17:
            short_sentences.append(s)

    for s in sentences_val:
        if len(s.tokens) > 29:
            long_sentences.append(s)

    X_normal, y_normal = tensor_from_sentence_pos(sentences)
    X_val_normal, y_val_normal = tensor_from_sentence_pos(sentences_val)

    X_easy, y_easy = tensor_from_sentence_pos(short_sentences)
    X_val_hard, y_val_hard = tensor_from_sentence_pos(long_sentences)

    return X_normal, y_normal, X_easy, y_easy, X_val_normal, y_val_normal, X_val_hard, y_val_hard
