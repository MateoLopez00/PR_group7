import numpy as np
from pyts.metrics import sakoe_chiba_band
from DTW_utils import _euclidean



def DTW(seq1, seq2, win_size=0.5):

    T1 = seq1.shape[0]
    T2 = seq2.shape[0]

    window = sakoe_chiba_band(T1, T2, win_size)

    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0.0

    # Compute the cost with the window
    for i in range(1, T1 + 1):
        j_min = max(1, i - window)
        j_max = min(T2, i + window)
        for j in range(j_min, j_max + 1):
            cost = _euclidean(seq1[i - 1], seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], 
                                 D[i, j - 1], 
                                 D[i - 1, j - 1])

    final_cost = D[T1, T2]

    return final_cost

def search(query, candidates, top_k):

    scores = []
    for seq in candidates:
        dist = DTW(query, seq)
        scores.append(dist)

    scores = scores.sort()
    return scores[:top_k]