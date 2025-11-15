import numpy as np
from pyts.metrics import sakoe_chiba_band
from DTW_utils import _euclidean



def DTW(seq1, seq2, win_size=0.5):
    """
    Calculates the DTW between two feature vectors
    Args:
        - seq1: the features of a query image
        - seq2: the features of every a candidate image
        Expects seq1 = extract_features(query_img)
    Returns: Cost matrix for these two images
    """
    T1 = seq1.shape[0]
    T2 = seq2.shape[0]
    # window[0, i] = j_start for given i 
    # window[1, i] = j_end for given i 
    window = sakoe_chiba_band(T1, T2, win_size)

    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0.0

    # Compute the cost with the window
    for i in range(1, T1 + 1):
        # get valid range for j 
        j_min = max(0,window[0,i-1]) + 1     
        j_max = min(T2, window[1,i-1]) 
     
        for j in range(j_min, j_max + 1):
            cost = _euclidean(seq1[i - 1], seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j],       #insertion
                                 D[i, j - 1],       #deletion
                                 D[i - 1, j - 1])   #match

    final_cost = D[T1, T2]

    return final_cost

def search(query, candidates, top_k):

    scores = []
    for seq in candidates:
        dist = DTW(query, seq)
        scores.append(dist)

    scores = scores.sort()
    return scores[:top_k]