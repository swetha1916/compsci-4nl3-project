# Imports
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Paths (Codabench standard)
input_dir = '/app/input'
output_dir = '/app/output/'
reference_dir = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')

score_file = os.path.join(output_dir, 'scores.json')
html_file = os.path.join(output_dir, 'detailed_results.html')


def write_file(file, content):
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)


def load_data():
    """Load ground truth and predictions, merge on ID."""
    # Ground truth
    y_true = pd.read_csv(os.path.join(reference_dir, 'test_labels.csv'))

    # Predictions
    y_pred = pd.read_csv(os.path.join(prediction_dir, 'predictions.csv'))

    # Merge on ID (VERY IMPORTANT)
    df = y_true.merge(y_pred, on="ID", suffixes=("_true", "_pred"))

    return df


def main():
    print("----------")
    print("Scoring program started")

    write_file(html_file, '<h1>Detailed Results</h1>')

    # Load merged dataframe
    df = load_data()

    # Extract labels
    y_true = df["Label_true"]
    y_pred = df["Label_pred"]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Macro: {f1_macro}")

    # Optional: duration
    metadata_path = os.path.join(prediction_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            duration = json.load(f).get('duration', -1)
    else:
        duration = -1

    # Scores dictionary
    scores = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "duration": duration
    }

    # Write scores.json (Codabench expects JSON)
    with open(score_file, "w") as f:
        json.dump(scores, f)

    # Write simple HTML feedback
    write_file(html_file, f"<p>Accuracy: {accuracy:.4f}</p>")
    write_file(html_file, f"<p>F1 Macro: {f1_macro:.4f}</p>")

    print("----------")
    print("Scoring program finished")
    print(scores)


if __name__ == "__main__":
    main()