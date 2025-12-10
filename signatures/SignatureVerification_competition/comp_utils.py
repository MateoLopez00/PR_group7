import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sign_utils import processFiles, DTW
import pandas as pd



def compare():
    writers = pd.read_csv("signatures/SignatureVerification_competition/writers.tsv", sep="\t", header=None)
    col_names = []
    col_names.append("Writer")

    for i in range(1,46):
        idx = f"{i:02d}"
        col_names.append(idx)

    df = pd.DataFrame(columns=col_names)
    for i, writer in enumerate(writers[0]):
        row = []
        writer = f"{writer:03d}"
        print(writer)
        features = ["x", "y", "pressure", "penup", "azimuth", "inclination", "v_x", "v_y"]
        enrollment = pd.read_csv(f"signatures/SignatureVerification_competition/enrollment/{writer}-g-01.tsv", sep ="\t", header=0)
        # Source - https://stackoverflow.com/a
        # Posted by Cina, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-12-08, License - CC BY-SA 4.0
        enrollment = enrollment[features]
        normalized_enrollment=(enrollment-enrollment.min())/(enrollment.max()-enrollment.min())
        seq1 = normalized_enrollment.select_dtypes(include=["number"]).values

        for j in range(1,46):
            idx = f"{j:02d}"
            verification = pd.read_csv(f"signatures/SignatureVerification_competition/verification/{writer}-{idx}.tsv", sep ="\t", header=0)
            verification = verification[features]
            normalized_verification=(verification-verification.min())/(verification.max()-verification.min())
            seq2 = normalized_verification.select_dtypes(include=["number"]).values
            score = DTW(seq1, seq2, 0.5)
            row.append(score)
        # Add row to df (does not yet work properly changed the indices of the loops maybe that helps)
        df.loc[i] = [writer] + row
    return df


if __name__ == "__main__":
    #processFiles("signatures\SignatureVerification_competition\enrollment")
    #processFiles("signatures\SignatureVerification_competition/verification")
    dissim = compare()
    dissim.to_csv("submission.csv")