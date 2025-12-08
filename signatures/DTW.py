from sign_utils import *
import pandas as pd

def compare():
    dissim = []
    writers = pd.read_csv("signatures/writers.tsv", sep="\t", header=None)

    for writer in writers[0]:
        writer = f"{writer:03d}"
        print(writer)
        features = ["x", "y", "pressure", "penup", "azimuth", "inclination", "v_x", "v_y"]
        enrollment = pd.read_csv(f"signatures/enrollment/{writer}-g-01.tsv", sep ="\t", header=0)
        # Source - https://stackoverflow.com/a
        # Posted by Cina, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-12-08, License - CC BY-SA 4.0
        enrollment = enrollment[features]
        normalized_enrollment=(enrollment-enrollment.min())/(enrollment.max()-enrollment.min())

        seq1 = normalized_enrollment.select_dtypes(include=["number"]).values
        for i in range(1,46):
            idx = f"{i:02d}"  # produces "01","02",...,"10",...,"45"
            verification = pd.read_csv(f"signatures/verification/{writer}-{idx}.tsv", sep ="\t", header=0)
            verification = verification[features]
            normalized_verification=(verification-verification.min())/(verification.max()-verification.min())
            seq2 = normalized_verification.select_dtypes(include=["number"]).values
            score = DTW(seq1, seq2, 0.5)
            dissim.append({'writer': writer, 'idx': idx, 'dissim': score})
    return pd.DataFrame(dissim)

if __name__ == "__main__":
    dissim = compare()
    dissim.to_csv("signatures\dissim.csv")
    print(dissim)