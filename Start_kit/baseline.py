import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# Load datasets
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")

# Labels
y_train = train["Label"]
y_val = val["Label"]

# Majority baseline
majority_label = train["Label"].mode()[0]
print("Majority label from training set:", majority_label)

majority_pred = [majority_label] * len(val)
print("Majority baseline (macrlo F1):", f1_score(y_val, majority_pred, average="macro"))

# Combine Post + Comment into one text string
X_train_text = train["Post"].fillna("") + " " + train["Comment"].fillna("")
X_val_text = val["Post"].fillna("") + " " + val["Comment"].fillna("")

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)
X_val_vec = vectorizer.transform(X_val_text)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_val_vec)
print("Logistic Regression (macro F1):", f1_score(y_val, lr_pred, average="macro"))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec.toarray(), y_train)
rf_pred = rf.predict(X_val_vec.toarray())
print("Random Forest (macro F1):", f1_score(y_val, rf_pred, average="macro"))

# Feedforward Neural Network
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    random_state=42
)
nn.fit(X_train_vec.toarray(), y_train)
nn_pred = nn.predict(X_val_vec.toarray())
print("Feedforward Neural Network (macro F1):", f1_score(y_val, nn_pred, average="macro"))
