import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("annotated.csv")

RANDOM_SEED = 42

# split train-temp 60/40
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=RANDOM_SEED)

# split validation-test 20/20
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

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