import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def create_knowledge_base(csv_path='legal_assistant_dataset.csv'):
    """Create a knowledge base from the legal dataset."""
    # Create knowledge base directory
    os.makedirs('knowledge_base', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Create case type knowledge base
    case_types = df['Case Type'].unique()
    case_type_kb = {}
    
    for case_type in case_types:
        case_type_data = df[df['Case Type'] == case_type]
        
        # Get common legal references for this case type
        legal_refs = case_type_data['Legal References'].value_counts().to_dict()
        
        # Get common verdicts for this case type
        verdicts = case_type_data['Verdict'].value_counts().to_dict()
        
        # Get common courts for this case type
        courts = case_type_data['Court'].value_counts().to_dict()
        
        # Get sample cases
        sample_cases = case_type_data.sample(min(5, len(case_type_data))).to_dict('records')
        
        case_type_kb[case_type] = {
            'description': f"Cases related to {case_type}",
            'common_legal_references': legal_refs,
            'common_verdicts': verdicts,
            'common_courts': courts,
            'sample_cases': sample_cases
        }
    
    # Save case type knowledge base
    with open('knowledge_base/case_types.json', 'w') as f:
        json.dump(case_type_kb, f, indent=4)
    
    # Create legal references knowledge base
    legal_refs = df['Legal References'].unique()
    legal_refs_kb = {}
    
    for legal_ref in legal_refs:
        legal_ref_data = df[df['Legal References'] == legal_ref]
        
        # Get common case types for this legal reference
        case_types = legal_ref_data['Case Type'].value_counts().to_dict()
        
        # Get common verdicts for this legal reference
        verdicts = legal_ref_data['Verdict'].value_counts().to_dict()
        
        # Get sample cases
        sample_cases = legal_ref_data.sample(min(5, len(legal_ref_data))).to_dict('records')
        
        legal_refs_kb[legal_ref] = {
            'description': f"Cases referencing {legal_ref}",
            'common_case_types': case_types,
            'common_verdicts': verdicts,
            'sample_cases': sample_cases
        }
    
    # Save legal references knowledge base
    with open('knowledge_base/legal_references.json', 'w') as f:
        json.dump(legal_refs_kb, f, indent=4)
    
    # Create judges knowledge base
    judges = df['Judge Name'].unique()
    judges_kb = {}
    
    for judge in judges:
        judge_data = df[df['Judge Name'] == judge]
        
        # Get common case types for this judge
        case_types = judge_data['Case Type'].value_counts().to_dict()
        
        # Get common verdicts for this judge
        verdicts = judge_data['Verdict'].value_counts().to_dict()
        
        # Get common courts for this judge
        courts = judge_data['Court'].value_counts().to_dict()
        
        # Get sample cases
        sample_cases = judge_data.sample(min(5, len(judge_data))).to_dict('records')
        
        judges_kb[judge] = {
            'description': f"Cases presided by {judge}",
            'common_case_types': case_types,
            'common_verdicts': verdicts,
            'common_courts': courts,
            'sample_cases': sample_cases
        }
    
    # Save judges knowledge base
    with open('knowledge_base/judges.json', 'w') as f:
        json.dump(judges_kb, f, indent=4)
    
    # Create courts knowledge base
    courts = df['Court'].unique()
    courts_kb = {}
    
    for court in courts:
        court_data = df[df['Court'] == court]
        
        # Get common case types for this court
        case_types = court_data['Case Type'].value_counts().to_dict()
        
        # Get common verdicts for this court
        verdicts = court_data['Verdict'].value_counts().to_dict()
        
        # Get common judges for this court
        judges = court_data['Judge Name'].value_counts().to_dict()
        
        # Get sample cases
        sample_cases = court_data.sample(min(5, len(court_data))).to_dict('records')
        
        courts_kb[court] = {
            'description': f"Cases heard in {court}",
            'common_case_types': case_types,
            'common_verdicts': verdicts,
            'common_judges': judges,
            'sample_cases': sample_cases
        }
    
    # Save courts knowledge base
    with open('knowledge_base/courts.json', 'w') as f:
        json.dump(courts_kb, f, indent=4)
    
    # Create FAQ knowledge base
    faqs = [
        {
            'question': 'What types of cases are most common?',
            'answer': f"Based on our data, the most common case types are: {', '.join(df['Case Type'].value_counts().head(3).index.tolist())}."
        },
        {
            'question': 'Which court handles the most cases?',
            'answer': f"Based on our data, {df['Court'].value_counts().index[0]} handles the most cases."
        },
        {
            'question': 'Who are the most active judges?',
            'answer': f"Based on our data, the most active judges are: {', '.join(df['Judge Name'].value_counts().head(3).index.tolist())}."
        },
        {
            'question': 'What are the most common legal references?',
            'answer': f"Based on our data, the most commonly cited legal references are: {', '.join(df['Legal References'].value_counts().head(3).index.tolist())}."
        },
        {
            'question': 'What is the most common verdict?',
            'answer': f"Based on our data, the most common verdict is: {df['Verdict'].value_counts().index[0]}."
        },
        {
            'question': 'How many cases are appealed?',
            'answer': f"Based on our data, {df['Appeal Status'].value_counts().get('Appeal Pending', 0) + df['Appeal Status'].value_counts().get('Appeal Dismissed', 0)} cases have been appealed."
        },
        {
            'question': 'What is the success rate of appeals?',
            'answer': f"Based on our data, the success rate of appeals is approximately {100 - (df['Appeal Status'].value_counts().get('Appeal Dismissed', 0) / (df['Appeal Status'].value_counts().get('Appeal Pending', 0) + df['Appeal Status'].value_counts().get('Appeal Dismissed', 0)) * 100):.2f}%."
        },
        {
            'question': 'What types of cases are most likely to be appealed?',
            'answer': f"Based on our data, {df[df['Appeal Status'] != 'No Appeal']['Case Type'].value_counts().index[0]} cases are most likely to be appealed."
        },
        {
            'question': 'What is the average time for a case to be resolved?',
            'answer': "Our data does not include case resolution time information."
        },
        {
            'question': 'Which judge has the highest rate of cases being appealed?',
            'answer': f"Based on our data, cases presided by {df[df['Appeal Status'] != 'No Appeal']['Judge Name'].value_counts().index[0]} are most likely to be appealed."
        }
    ]
    
    # Save FAQ knowledge base
    with open('knowledge_base/faqs.json', 'w') as f:
        json.dump(faqs, f, indent=4)
    
    # Create a vectorized representation of all cases for similarity search
    print("Creating vector database for similarity search...")
    
    # Combine relevant columns for search
    df['search_text'] = df['Case Type'] + ' ' + df['Court'] + ' ' + df['Judge Name'] + ' ' + df['Case Summary'] + ' ' + df['Key Arguments'] + ' ' + df['Legal References'] + ' ' + df['Verdict']
    
    # Preprocess text
    df['processed_search_text'] = df['search_text'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_search_text'])
    
    # Save vectorizer and matrix
    import pickle
    with open('knowledge_base/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('knowledge_base/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Save processed dataframe for search
    df.to_pickle('knowledge_base/processed_cases.pkl')
    
    print("Knowledge base creation complete!")

if __name__ == "__main__":
    create_knowledge_base() 