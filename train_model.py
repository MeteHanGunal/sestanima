import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('7allV03.csv') 

texts = data['text']  
labels = data['category'] 

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

with open('turkish_text_classification_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model başarıyla kaydedildi.")
