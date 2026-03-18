import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Majority label
majority_label = train["Label"].mode()[0]

# Predict
preds = pd.DataFrame({
    "ID": test["ID"],
    "Label": [majority_label] * len(test)
})

preds.to_csv("predictions.csv", index=False)
