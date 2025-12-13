import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sign_utils import processFiles, DTW
import pandas as pd
import numpy as np



def compare(win_size=0.2, ds=1):
    writers = pd.read_csv("./writers.tsv", sep="\t", header=None)
    col_names = []
    col_names.append("Writer")

    for i in range(1,46):
        idx = f"{i:02d}"
        col_names.append(idx)

    df = pd.DataFrame(columns=col_names)
    eps = 1e-8

    for i, writer in enumerate(writers[0]):
        row = []
        writer = f"{int(writer):03d}"
        print(writer)
        features = ["x", "y", "pressure", "penup", "azimuth", "inclination", "v_x", "v_y"]
        enrollment = pd.read_csv(f"./enrollment/{writer}-g-01.tsv", sep ="\t", header=0)
        # Source - https://stackoverflow.com/a
        # Posted by Cina, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-12-08, License - CC BY-SA 4.0
        enrollment = enrollment[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        en_min = enrollment.min()
        en_rng = (enrollment.max() - enrollment.min()).replace(0, eps)

        seq1 = ((enrollment - en_min) / en_rng).values
        if ds > 1:
            seq1 = seq1[::ds]

        for j in range(1,46):
            idx = f"{j:02d}"
            verification = pd.read_csv(f"./verification/{writer}-{idx}.tsv", sep ="\t", header=0)
            verification = verification[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            seq2 = ((verification - en_min) / en_rng).values
            if ds > 1:
                seq2 = seq2[::ds]

            score = DTW(seq1, seq2, win_size)
            row.append(score)
        # Add row to df 
        df.loc[i] = [writer] + row
    return df


if __name__ == "__main__":
    #processFiles("signatures\SignatureVerification_competition\enrollment")
    #processFiles("signatures\SignatureVerification_competition/verification")
    dissim = compare(win_size=0.2, ds=1)
    dissim.to_csv("submission.csv", index=False)
