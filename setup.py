from setuptools import setup, find_packages

setup(
    name="legal-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "requests",
    ],
    author="AI Developer",
    author_email="developer@example.com",
    description="A legal assistant chatbot powered by AI",
    keywords="legal, assistant, chatbot, AI, NLP",
    url="https://github.com/yourusername/legal-assistant",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
) 