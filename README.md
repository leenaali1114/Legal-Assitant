# Legal Assistant AI

An intelligent legal assistant chatbot powered by machine learning and natural language processing. This application analyzes legal cases, provides insights, and assists with legal queries.

## Features

- **AI-Powered Chat**: Interact with the legal assistant through a conversational interface
- **Case Search**: Find similar cases based on descriptions
- **Case Prediction**: Predict case types, verdicts, and relevant legal references
- **Legal Analytics**: Visualize trends and patterns in legal data
- **Knowledge Base**: Access information about case types, legal references, judges, and courts

## Project Overview

The Legal Assistant AI is designed to help legal professionals, law students, and individuals seeking legal information. It uses machine learning models trained on a dataset of legal cases to provide insights, predictions, and relevant information.

The system consists of several components:
- Data analysis module for extracting insights from legal cases
- Machine learning models for case type, verdict, and legal reference prediction
- Knowledge base creation for storing structured legal information
- Web interface with chat, search, prediction, and analytics features
- Integration with Groq API for advanced natural language understanding

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/legal-assistant.git
   cd legal-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Groq API key: `GROQ_API_KEY=your_api_key_here`
   - Add a secret key: `SECRET_KEY=your_secret_key_here`

## Dataset

The system requires a legal case dataset in CSV format. The dataset should include the following columns:
- Case ID
- Case Type
- Court
- Judge Name
- Date
- Plaintiff
- Defendant
- Case Summary
- Key Arguments
- Verdict
- Appeal Status
- Legal References
- Outcome Summary

Place your dataset file named `legal_assistant_dataset.csv` in the project root directory.

## Running the Project

To run the application, simply execute:

```
python run.py
```

This will start the Flask web application. Open your browser and navigate to `http://localhost:5000` to access the legal assistant interface.

### Alternative Startup Methods

You can also start the application directly with:

```
python legal_assistant_chatbot.py
```

### Setup and Preparation

Before using the application for the first time, you should prepare the data, models, and knowledge base. Run these steps in sequence:

1. Data Analysis:
   ```
   python legal_data_analysis.py
   ```

2. Train Models:
   ```
   python train_legal_model.py
   ```

3. Create Knowledge Base:
   ```
   python create_knowledge_base.py
   ```

## Using the Legal Assistant

Once the application is running, you can:

1. **Chat with the Assistant**: Ask legal questions and get information about cases, laws, and legal concepts.

2. **Search for Similar Cases**: Describe a legal situation to find similar cases in the database.

3. **Get Case Predictions**: Provide details about a case to get predictions on case type, verdict, and relevant legal references.

4. **Explore Analytics**: View visualizations and insights about the legal case database.

For examples of questions you can ask the Legal Assistant AI, see the [Sample Queries](sample_queries.md) document.

## Project Structure

- `legal_data_analysis.py`: Analyzes the legal dataset and generates visualizations
- `train_legal_model.py`: Trains machine learning models on the legal dataset
- `create_knowledge_base.py`: Creates a knowledge base from the legal dataset
- `legal_assistant_chatbot.py`: Main application file with Flask server
- `run.py`: Script to run the application
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and image files
- `models/`: Trained machine learning models
- `knowledge_base/`: JSON files containing legal knowledge
- `sample_queries.md`: Examples of questions to ask the assistant

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **API Integration**: Groq API for advanced language model capabilities

## Customization

You can customize the application by:
- Using your own legal dataset (following the required format)
- Modifying the machine learning models in `train_legal_model.py`
- Extending the knowledge base creation in `create_knowledge_base.py`
- Adding new features to the web interface in the templates and static files

## Troubleshooting

- If you encounter issues with NLTK resources, run the following commands:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

- If the application fails to start, check that all required files and directories exist and that your `.env` file contains the necessary API keys.

- For model training issues, ensure your dataset has sufficient examples of each case type, verdict, and legal reference.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 