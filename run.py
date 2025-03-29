import os
import sys

def check_requirements():
    """Check if all required files and directories exist."""
    required_files = [
        'legal_assistant_dataset.csv',
        'legal_data_analysis.py',
        'train_legal_model.py',
        'create_knowledge_base.py',
        'legal_assistant_chatbot.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file '{file}' not found.")
            return False
    
    return True

def setup_environment():
    """Set up the environment for the application."""
    # Create required directories
    os.makedirs('static/analysis', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('knowledge_base', exist_ok=True)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Creating a template .env file.")
        with open('.env', 'w') as f:
            f.write("# API Keys\n")
            f.write("GROQ_API_KEY=your_groq_api_key_here\n\n")
            f.write("# Flask Configuration\n")
            f.write("SECRET_KEY=your_secret_key_here\n")
            f.write("FLASK_ENV=development\n")
        print("Please edit the .env file with your actual API keys.")

def start_application():
    """Start the Flask application."""
    print("Starting the legal assistant application...")
    os.system('python legal_assistant_chatbot.py')

def main():
    """Main function to run the application."""
    print("Legal Assistant AI")
    print("=================")
    
    if not check_requirements():
        sys.exit(1)
    
    setup_environment()
    
    # Directly start the application
    start_application()

if __name__ == "__main__":
    main() 