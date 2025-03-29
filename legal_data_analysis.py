import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_legal_data(csv_path='legal_assistant_dataset.csv'):
    """Analyze the legal dataset and generate insights."""
    # Create output directory for visualizations
    os.makedirs('static/analysis', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Count of case types
    case_type_counts = df['Case Type'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=case_type_counts.index, y=case_type_counts.values)
    plt.title('Distribution of Case Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/case_types.png')
    
    # Count of courts
    court_counts = df['Court'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=court_counts.index, y=court_counts.values)
    plt.title('Distribution of Courts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/courts.png')
    
    # Count of judges
    judge_counts = df['Judge Name'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=judge_counts.index, y=judge_counts.values)
    plt.title('Top 10 Judges by Case Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/judges.png')
    
    # Verdict distribution
    verdict_counts = df['Verdict'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=verdict_counts.index, y=verdict_counts.values)
    plt.title('Distribution of Verdicts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/verdicts.png')
    
    # Appeal status distribution
    appeal_counts = df['Appeal Status'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=appeal_counts.index, y=appeal_counts.values)
    plt.title('Distribution of Appeal Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/appeals.png')
    
    # Legal references distribution
    reference_counts = df['Legal References'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=reference_counts.index, y=reference_counts.values)
    plt.title('Distribution of Legal References')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/analysis/references.png')
    
    # Cases over time
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    year_counts = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values)
    plt.title('Cases Over Time (by Year)')
    plt.tight_layout()
    plt.savefig('static/analysis/cases_over_time.png')
    
    # Case type by court heatmap
    case_court_counts = pd.crosstab(df['Case Type'], df['Court'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(case_court_counts, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Case Types by Court')
    plt.tight_layout()
    plt.savefig('static/analysis/case_court_heatmap.png')
    
    # Generate summary statistics
    summary = {
        'total_cases': len(df),
        'case_types': len(case_type_counts),
        'courts': len(court_counts),
        'judges': len(df['Judge Name'].unique()),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'top_case_type': case_type_counts.index[0],
        'top_court': court_counts.index[0],
        'top_judge': judge_counts.index[0],
        'top_verdict': verdict_counts.index[0],
    }
    
    return summary

if __name__ == "__main__":
    summary = analyze_legal_data()
    print("Analysis complete. Visualizations saved to 'static/analysis/' directory.")
    print(f"Summary: {summary}") 