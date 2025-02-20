import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 

from config import BASE_DIR, DATA_PATH


start_time = time.time()
# Read data into a Pandas DataFrame
df = pd.read_csv(BASE_DIR / DATA_PATH / "emails" / "email.csv")
# Select features (X) and labels (y)
X = df['Message']
y = df['Category']

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

end_time: float = time.time()
execution_time: float = end_time - start_time
print(f"Execution time: {execution_time} milliseconds")
