# library imports
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


# load datasets
train = pd.read_csv("train.csv")
val = pd.read_csv("validation.csv")
test = pd.read_csv("test.csv")


# feature set and target
X_train = train[["Post", "Comment"]]
y_train = train["Label"]

X_val = val[["Post", "Comment"]]
y_val = val["Label"]

X_test = test[["Post", "Comment"]]
y_test = test["Label"]


# majority label (baseline)
majority_label = train["Label"].mode()[0]
print("Majority label from training set:", majority_label)

base_pred = [majority_label] * len(val)
print("Majority label", f1_score(y_val, base_pred, average="weighted"))


# combine Post + Comment into one string per row
X_train_text = X_train["Post"] + " " + X_train["Comment"]
X_val_text   = X_val["Post"] + " " + X_val["Comment"]


# convert text to TF-IDF features
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train_text)
X_val_vec = vectorizer.transform(X_val_text)


# logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

lr_pred = lr.predict(X_val_vec)
print("Logistic regression", f1_score(y_val, lr_pred, average="weighted"))
# average="weighted": since we have 3 classes, the f1 score must combine the f1 of each class into one number


# random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec.toarray(), y_train)

rf_pred = rf.predict(X_val_vec.toarray())
print("Random Forest:", f1_score(y_val, rf_pred, average="weighted"))


# feedforward neural network
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=10,
    random_state=42
)
nn.fit(X_train_vec.toarray(), y_train)

nn_pred = nn.predict(X_val_vec.toarray())
print("Neural Network:", f1_score(y_val, nn_pred, average="weighted"))

# Majority label: 0.5547854785478548
# Logistic regression 0.5923483271088784
# Random Forest: 0.6013472828541321         (best)
# Neural Network: 0.5622014925373134
