import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

label_dict = {'DESC': 0, 'ENTY': 1, 'ABBR': 2, 'HUM': 3, 'NUM': 4, 'LOC': 5}
def read_data():
    train_data, train_label, test_data, test_label = [], [], [], []
    max_len = 0
    with open('data/train.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(':')
            train_label.append(label_dict[line[0]])
            sent = line[1].split(' ')[1:]
            # print(sent)
            train_data.append(sent)
            if len(sent) > max_len:
                max_len = len(sent)
    with open('data/test.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(':')
            test_label.append(label_dict[line[0]])
            test_data.append(line[1].split(' '))

    train_label = to_categorical(train_label, 6)
    test_label = to_categorical(test_label, 6)
    print(len(train_data), len(train_label), len(test_data), len(test_label), max_len)
    return train_data, train_label, test_data, test_label, max_len

def read_data_mr():
    data = []
    label = []
    max_len = 0
    with open('data/rt-polaritydata/rt-polarity.pos', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data.append(line)
            if len(line) > max_len:
                max_len = len(line)
            label.append(1)
    with open('data/rt-polaritydata/rt-polarity.neg', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data.append(line)
            if len(line) > max_len:
                max_len = len(line)
            label.append(0)
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True)
    print(len(train_data), len(train_label), len(test_data), len(test_label), max_len)
    train_label = to_categorical(train_label, 2)
    test_label = to_categorical(test_label, 2)
    return train_data, train_label, test_data, test_label, max_len

def get_vocab(train_data):
    vocab = {}
    vocab['pad'] = 0
    index = 1
    for s in train_data:
        for w in s:
            if w.lower() not in vocab.keys():
                vocab[w.lower()] = index
                index += 1
    vocab['unk'] = index
    print(index)
    print(len(vocab.keys()))
    # print(vocab)
    return vocab

def get_embedding_matrix(vocab):
    import gensim
    done_set = set()
    embedding_matrix = np.zeros(shape=[len(vocab), 300], dtype=np.float32)
    path = 'GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    word2vec_vocab = word2vec.vocab
    for word in word2vec_vocab.keys():
        if word in vocab.keys() and not word in done_set:
            done_set.add(word)
            embedding_matrix[vocab[word]] = word2vec[word]
    # with open('glove.840B.300d.txt', 'r', encoding='utf8') as f:
    #     lines = f.readlines()
    #     # print(str(lines[1]).encode('utf8'))
    #     for line in lines:
    #         try:
    #             vector = line.strip().split()
    #             if len(vector) != 301:
    #                 continue
    #             else:
    #                 word = line.split()[0]
    #         except:
    #             continue
    #         if word in vocab.keys() and not word in done_set:
    #             done_set.add(word)
    #             embedding_matrix[vocab[word]] = np.array(vector[1:])
    oov_words = vocab.keys() - list(done_set)
    print('oov rate: ', len(oov_words) / len(vocab))
    for oov in oov_words:
        embedding_matrix[vocab[oov]] = np.random.uniform(low=-0.01, high=0.01, size=[1, 300])
    np.save('word2vec_terc', embedding_matrix)

def get_train_test_data(vocab, train_data, test_data, max_len):
    x_train = []
    x_test = []
    for s in train_data:
        sent = []
        for w in s:
            w = w.lower()
            sent.append(vocab[w])
        length = len(sent)
        if length < max_len:
            for i in range(max_len-length):
                sent.append(0)
        x_train.append(sent)

    for s in test_data:
        sent = []
        for w in s:
            w = w.lower()
            if w in vocab.keys():
                sent.append(vocab[w])
            else:
                sent.append(vocab['unk'])
        length = len(sent)
        if length < max_len:
            for i in range(max_len-length):
                sent.append(0)
        x_test.append(sent)

    print(x_test[:2])
    return x_train, x_test

def main(data='terc'):
    if data == 'MR':
        train_data, train_label, test_data, test_label, max_len = read_data_mr()
        vocab = get_vocab(train_data)
        # get_embedding_matrix(vocab)
        x_train, x_test = get_train_test_data(vocab, train_data, test_data, max_len)
    else:
        train_data, train_label, test_data, test_label, max_len = read_data()
        vocab = get_vocab(train_data)
        # get_embedding_matrix(vocab)
        x_train, x_test = get_train_test_data(vocab, train_data, test_data, max_len)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    # print(train_label[:100])
    return x_train, train_label, x_test, test_label, max_len, len(vocab)
#
# if __name__ == '__main__':
    # x_train, train_label, x_test, test_label, max_len, vocab_size = main()


