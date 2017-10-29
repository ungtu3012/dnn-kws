import os
import utils
import random
import numpy as np

skip_step = 50
stop_step = 100
root = 'dataset/'

def get_words():
    for word in os.listdir(root):
        path = os.path.join(root, word)
        if os.path.isdir(path) and '_' not in word:
            yield word

words = list(get_words())

X, Y, X_train, X_test, Y_train, Y_test = [], [], [], [], [], []

def paths(root):
    count = 0
    for path, _, files in os.walk(root):
        for file in files:
            if file.endswith('.wav'):
                wave_path = os.path.join(path, file)
                yield wave_path
                count += 1
                if count % skip_step == 0:
                    print('Loaded %d data, current path %s' % (count, wave_path))
                if count == stop_step:
                    print('Loaded %d data, stop' % count)
                    return

def reload():
    X.clear(), Y.clear()

def load(word, path=root):
    for wave_path in paths(path+word):
        X.append(utils.vectorize_x(wave_path))
        Y.append(words.index(word))

def shuffle():
    global X
    global Y
    num = len(X)
    indices = list(range(num))
    random.shuffle(indices)
    X = np.array([X[i] for i in indices])
    Y = [Y[i] for i in indices]

def split():
    global X_train
    global X_test
    global Y_train
    global Y_test
    num = len(X)
    percent_train = 0.8
    num_train = int(num * percent_train)
    X_train = np.array(X[:num_train])
    X_test = np.array(X[num_train:])
    Y_train = Y[:num_train]
    Y_test = Y[num_train:]
    print(len(X_train), len(Y_train), len(X_test), len(Y_test))

def export(file_name, X=X, Y=Y):
    print('[+] Exporting %s_x.npy' % file_name)
    np.save('%s_x.npy' % file_name, np.array(X))
    print('[+] Exporting %s_y.npy' % file_name)
    np.save('%s_y.npy' % file_name, np.array(Y))
    print('[+] Done!')

def main():

    for word in words:
        print('[+] Prepare "%s" set, index %d' % (word, words.index(word)))
        load(word)

    print('[+] Shuffle dataset')
    shuffle()

    print('[+] Split train/test set')
    split()

    print('[+] Exporting')
    export('train', X=X_train, Y=Y_train)
    export('test', X=X_test, Y=Y_test)

    print('Done!')

if __name__ == '__main__':
    main()