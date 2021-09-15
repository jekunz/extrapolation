from load_ud import read_conllu
import pickle, random

random.seed(42)

path = 'en_ewt-ud-train.conllu'
path_dev = 'en_ewt-ud-dev.conllu'


# select 1000 random sentences from .conllu
# save them to file as sentence objects
def save_to_file(path, layer, file_name):
    sentences = read_conllu(layer, path)

    # generator to list
    s_train = []
    for s in sentences:
        s_train.append(s)
    print("Data for {0} loaded; contains {1} sentences".format(file_name, len(s_train)))

    if len(s_train) > 1000:
        s_select = random.sample(s_train, 1000)
    else: s_select = sentences

    with open(file_name, 'wb') as train_file:
        pickle.dump(s_select, train_file)


for i in range(0,13):
    print('Layer: {0} '.format(i))
    file_train = 'pickle2/train{0}.sentences'.format(i)
    file_dev = 'pickle2/dev{0}.sentences'.format(i)
    save_to_file(path, i, file_train)
    save_to_file(path_dev, i, file_dev)