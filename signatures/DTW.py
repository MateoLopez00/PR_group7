from sign_utils import *
import pandas as pd

def compare():
    dissim = []
    writers = pd.read_csv("signatures/writers.tsv", sep="\t", header=None)

    for writer in writers[0]:
        writer = f"{writer:03d}"
        enrollment = pd.read_csv(f"signatures/enrollment/{writer}-g-01.tsv", sep ="\t", header=0)
        seq1 = enrollment.select_dtypes(include=["number"]).values
        for i in range(1,46):
            print(i)
            idx = f"{i:02d}"  # produces "01","02",...,"10",...,"45"
            verification = pd.read_csv(f"signatures/verification/{writer}-{idx}.tsv", sep ="\t", header=0)
            seq2 = verification.select_dtypes(include=["number"]).values
            score = DTW(seq1, seq2, 0.5)
            dissim.append({'writer': writer, 'idx': idx, 'dissim': score})
    return pd.DataFrame(dissim)

if __name__ == "__main__":
    dissim = compare()
    print(dissim)