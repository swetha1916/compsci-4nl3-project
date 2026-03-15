import pandas as pd
from sklearn.metrics import accuracy_score

# train 60%
train_df = pd.read_csv("train.csv")

# validation 20%
val_df = pd.read_csv("validation.csv")

# test 20%
test_df = pd.read_csv("test.csv")

# find majority label
majority_label = train_df["Label"].mode()[0]
print("Majority label from training set:", majority_label)

# predictions
val_preds = [majority_label] * len(val_df)
test_preds = [majority_label] * len(test_df)

# accuracy
val_accuracy = accuracy_score(val_df["Label"], val_preds)
test_accuracy = accuracy_score(test_df["Label"], test_preds)

print(f"Validation accuracy: {val_accuracy:.6f}")
print(f"Test accuracy: {test_accuracy:.6f}")
