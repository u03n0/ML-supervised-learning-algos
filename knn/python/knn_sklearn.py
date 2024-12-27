from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 


# Read data into a Pandas DataFrame
df = pd.read_csv("../data/emails/email.csv")
# Select features (X) and labels (y)
X = df['Message']
y = df['Category']

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))



