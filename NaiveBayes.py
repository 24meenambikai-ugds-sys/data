import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("spam.csv",encoding = "latin")
print("Dataset sample:\n", df.head())
print("\nShape:", df.shape)
print("\nClass distribution:\n", df["label"].value_counts())


df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words="english")
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))
print("Sample feature names:", vectorizer.get_feature_names_out()[:10])


model = MultinomialNB()
model.fit(X_train_counts, y_train)
y_pred = model.predict(X_test_counts)


print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))

new_message = ["Congratulations! You have won a free prize, claim now by calling this number"]
new_message_counts = vectorizer.transform(new_message)
prediction = model.predict(new_message_counts)
print("\nNew message prediction:", "spam" if prediction[0] == 1 else "ham")


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SMS Spam Detection (Naive Bayes)")
plt.tight_layout()
plt.show()

