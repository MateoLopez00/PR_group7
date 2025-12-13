import numpy as np
from pyts.metrics import sakoe_chiba_band
import pandas as pd
from pathlib import Path


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Simple Euclidean distance; DTW.py imports this."""
    return float(np.linalg.norm(a - b))

def DTW(seq1, seq2, win_size=0.5):
    """
    Calculates the DTW between two feature vectors
    Args:
        - seq1: the features of a query image
        - seq2: the features of every a candidate image
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

def computeVelocity(file_path):

    cols = ['t', 'x', 'y', 'pressure', 'penup', 'azimuth', 'inclination']
    data = pd.read_csv(file_path, sep="\t", header=None, names=cols)
    data = data.apply(pd.to_numeric, errors='coerce')

    data = data.sort_values(data.columns[0])
    
    # time difference
    d_t = data["t"].diff().fillna(0.0)
    
    # position differences
    d_x = data['x'].diff().fillna(0.0)
    d_y = data['y'].diff().fillna(0.0)

    # avoid division by zero: replace 0 dt with NaN, then fill velocities with 0
    dt_safe = d_t.replace(0.0, np.nan)
    v_x = d_x.div(dt_safe).fillna(0.0)
    v_y = d_y.div(dt_safe).fillna(0.0)

    # if penup is binary (1 = pen lifted), zero velocities when pen is up
    if set(data['penup'].unique()).issubset({0, 1}):
        v_x = v_x.where(data['penup'] == 0, 0.0)
        v_y = v_y.where(data['penup'] == 0, 0.0)

    data['v_x'] = v_x
    data['v_y'] = v_y

    out_cols = cols + ['v_x', 'v_y']

    # overwrite original file (tab-separated, no header, same column order + new cols at end)
    data.to_csv(file_path, sep="\t", header=out_cols, index=False, float_format='%.6f')

    return data

def processFiles(folder_path):
    folder = Path(folder_path)

    for file in folder.iterdir():
        computeVelocity(file)

if __name__ == "__main__":
    processFiles("signatures/verification")
