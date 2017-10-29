import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
num_classes = ord('z') - ord('a') + 1 + 1 + 1

def mfcc(file_name):
    wave, sr = librosa.load(file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    return np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)

def unvectorize_y(y):
    y = np.asarray([SPACE_TOKEN if i==0 else chr(FIRST_INDEX + i) for i in y])
    return ''.join(y).replace(SPACE_TOKEN, ' ').replace('{', '_')

def decode(d):
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    return str_decoded

def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def vectorize_x(path):
    x = mfcc(path)
    x = (x - np.mean(x)) / np.std(x)
    x = x.reshape([-1, 1])
    return x

def vectorize_y(y):
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
    y = np.hstack([SPACE_TOKEN if c == ' ' else list(c) for c in y])
    y = np.asarray([SPACE_INDEX if c == SPACE_TOKEN else ord(c) - FIRST_INDEX for c in y])
    y = np.asarray([c if c >=0 else 0 for c in y])
    return y

def plot_x(x):
    plt.plot(x)
    plt.show()

def main():
    path = 'dataset/yes/0a7c2a8d_nohash_0.wav'
    x = vectorize_x(path)
    plot_x(x)

if __name__ == '__main__':
    main()
