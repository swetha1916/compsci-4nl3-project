import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data.csv")
df["ID"] = df["ID"].astype(int)

# First split: train (60%) vs temp (40%)
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df["Label"], random_state=42
)

# Second split: validation (20%) vs test (20%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=42
)

# Save splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df_nolabel = test_df.drop(columns=["Label"])
test_df_nolabel.to_csv("test.csv", index=False)

test_df[["ID", "Label"]].to_csv("test_labels.csv", index=False)