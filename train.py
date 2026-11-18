"""Train the fake-news classifier and save ONE file: model.joblib (vectorizer + model together)."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
fake["label"] = 0   # 0 = fake
true["label"] = 1   # 1 = real

df = pd.concat([fake, true], ignore_index=True)
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)

# Drop empty rows and exact duplicates; otherwise the same article lands in both
# train and test and inflates the score.
df = df[df["content"].str.len() > 20]
df = df.drop_duplicates(subset="content").reset_index(drop=True)
print(f"Training rows after cleaning: {len(df)}\n{df['label'].value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000,
                              ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=5)),
])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["fake", "real"]))

joblib.dump(pipe, "model.joblib", compress=3)
print("Saved model.joblib")
