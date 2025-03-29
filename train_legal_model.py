import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text data for NLP."""
    # Tokenize
    tokens = nltk.word_tokenize(str(text).lower())
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    return ' '.join(tokens)

def train_legal_model(csv_path='legal_assistant_dataset.csv'):
    """Train NLP models on the legal dataset."""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Combine text features for input
    df['combined_text'] = df['Case Type'] + ' ' + df['Court'] + ' ' + df['Case Summary'] + ' ' + df['Key Arguments']
    
    # Preprocess text
    print("Preprocessing text data...")
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # Prepare features and targets
    X = df['processed_text']
    
    # Train case type prediction model
    print("Training case type prediction model...")
    case_type_encoder = LabelEncoder()
    y_case_type = case_type_encoder.fit_transform(df['Case Type'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_case_type, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train model
    case_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
    case_type_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = case_type_model.predict(X_test_tfidf)
    print("Case Type Prediction Performance:")
    print(classification_report(y_test, y_pred, target_names=case_type_encoder.classes_))
    
    # Train verdict prediction model
    print("Training verdict prediction model...")
    verdict_encoder = LabelEncoder()
    y_verdict = verdict_encoder.fit_transform(df['Verdict'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_verdict, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization (reusing the same vectorizer)
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train model
    verdict_model = RandomForestClassifier(n_estimators=100, random_state=42)
    verdict_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = verdict_model.predict(X_test_tfidf)
    print("Verdict Prediction Performance:")
    print(classification_report(y_test, y_pred, target_names=verdict_encoder.classes_))
    
    # Train legal reference prediction model
    print("Training legal reference prediction model...")
    reference_encoder = LabelEncoder()
    y_reference = reference_encoder.fit_transform(df['Legal References'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_reference, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization (reusing the same vectorizer)
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train model
    reference_model = RandomForestClassifier(n_estimators=100, random_state=42)
    reference_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = reference_model.predict(X_test_tfidf)
    print("Legal Reference Prediction Performance:")
    print(classification_report(y_test, y_pred, target_names=reference_encoder.classes_))
    
    # Save models and encoders
    print("Saving models and encoders...")
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    with open('models/case_type_model.pkl', 'wb') as f:
        pickle.dump(case_type_model, f)
    
    with open('models/case_type_encoder.pkl', 'wb') as f:
        pickle.dump(case_type_encoder, f)
    
    with open('models/verdict_model.pkl', 'wb') as f:
        pickle.dump(verdict_model, f)
    
    with open('models/verdict_encoder.pkl', 'wb') as f:
        pickle.dump(verdict_encoder, f)
    
    with open('models/reference_model.pkl', 'wb') as f:
        pickle.dump(reference_model, f)
    
    with open('models/reference_encoder.pkl', 'wb') as f:
        pickle.dump(reference_encoder, f)
    
    print("Model training complete!")

if __name__ == "__main__":
    train_legal_model() 