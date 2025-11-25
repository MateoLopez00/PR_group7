import numpy as np
from DTW_utils import _euclidean


def _sakoe_chiba_band(t1: int, t2: int, win_size: float):
    """Lightweight Sakoe–Chiba band approximation without pyts.

    Args:
        t1: Length of the first sequence.
        t2: Length of the second sequence.
        win_size: Fractional window size (0 < win_size <= 1).

    Returns:
        A (2, t1) integer array where ``band[0, i]`` and ``band[1, i]``
        give the inclusive start/end indices on the second sequence that
        are permitted for position ``i`` of the first sequence. Indices
        are zero-based, matching the expectations of the DTW loop below.
    """
    if not (0 < win_size <= 1):
        raise ValueError("win_size must be in (0, 1]")

    radius = int(np.ceil(win_size * max(t1, t2)))
    centers = np.linspace(0, t2 - 1, t1)
    starts = np.clip(np.floor(centers - radius), 0, t2 - 1).astype(int)
    ends = np.clip(np.ceil(centers + radius), 0, t2 - 1).astype(int)
    return np.vstack([starts, ends])



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
    window = _sakoe_chiba_band(T1, T2, win_size)

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

def search(query_feat, validation_candidates, top_k=None):
    results = []

    for entry in validation_candidates:
        cand_feat = entry["features"]
        dist = DTW(query_feat, cand_feat)
        results.append((dist, entry["loc"], entry["word"], entry["features"]))

    results.sort(key=lambda x: x[0])
    if top_k is not None:
        results = results[:top_k]
    return results
