import numpy as np


def avgdata(data):
    n = data.shape[0]
    k = data.shape[1]
    for i in range(n):
        for j in range(0, k, 3):
            sumt = np.sum(data[i,j:j + 3])
            data[i, j:j + 3] = sumt / 3
    return data

def avgseq(seq):
    n = len(seq)
    for i in range(0, n, 100):
        sumt = np.sum(seq[i:i + 100])
        seq[i:i + 100] = round(sumt / 100)
    return seq

