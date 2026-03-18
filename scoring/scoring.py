import os
import json
import pandas as pd
from sklearn.metrics import f1_score

# a small help function leads to the submission file
def find_csv_file(folder):
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            return os.path.join(folder, file_name)
    raise FileNotFoundError(f"No file in {folder}")


#https://docs.codabench.org/dev/Organizers/Benchmark_Creation/Competition-Bundle-Structure/?utm_source=chatgpt.com
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"

    reference_dir = os.path.join(input_dir, "ref")# the hidden values
    submission_dir = os.path.join(input_dir, "res")# where participants submitted predictions

    print("scoring started")
    print("input_dir exists:", os.path.exists("/app/input"))
    print("ref dir:", os.listdir("/app/input/ref") if os.path.exists("/app/input/ref") else "missing")
    print("res dir:", os.listdir("/app/input/res") if os.path.exists("/app/input/res") else "missing")

    os.makedirs(output_dir, exist_ok=True)

    reference_file = find_csv_file(reference_dir)
    ref_df = pd.read_csv(reference_file)

    submission_file = find_csv_file(submission_dir)
    sub_df = pd.read_csv(submission_file)

    #check tne required columns
    required_columns = {"ID", "Label"}

    if not required_columns.issubset(ref_df.columns):
        raise ValueError("Reference file must contain columns: ID and Label")
    if not required_columns.issubset(sub_df.columns):
        raise ValueError("Submission file must contain columns: ID and Label")

    ref_df = ref_df[["ID", "Label"]].copy()
    sub_df = sub_df[["ID", "Label"]].copy()

    ref_df = ref_df.rename(columns={"Label": "true_label"})
    sub_df = sub_df.rename(columns={"Label": "pred_label"})

    #merge by id
    merged = pd.merge(ref_df, sub_df, on="ID", how="left")

    #check for missing prediction
    if merged["pred_label"].isna().any():
        missing_ids = merged.loc[merged["pred_label"].isna(), "ID"].tolist()
        raise ValueError(f"Missing predictions for IDs: {missing_ids[:10]}")

    #check if there are extra id by comparing counts
    if len(sub_df) != len(ref_df):
        raise ValueError("Submission must contain exactly one prediction for each test ID")

    #weight f1
    f1 = f1_score(merged["true_label"], merged["pred_label"], average="weighted")

    scores = {
        "f1": float(f1)
    }

    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    print(f"f1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
