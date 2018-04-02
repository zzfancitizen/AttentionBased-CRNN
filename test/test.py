import numpy as np


def map_to_sequence(input):
    seq_vectors = []
    shape = np.shape(input)
    for i in range(shape[1]):
        col_vectors = input[:, i, :]
        col_vector = np.concatenate(col_vectors, axis=0)
        seq_vectors.append(col_vector)
    seq_vectors = np.reshape(seq_vectors, (shape[1] * shape[2], shape[0]))
    return seq_vectors


if __name__ == '__main__':
    input = np.random.rand(16, 16, 512)
    sequence = map_to_sequence(input)
    print(sequence)
    print(sequence.shape)
