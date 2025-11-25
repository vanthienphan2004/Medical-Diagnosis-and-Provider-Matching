import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from model_train import DiagnosticModel
from recommender import RecommenderEngine
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def visualize_system():
    """
    Creates comprehensive visualizations of the NeuroTriage system.
    """
    
    # Initialize model
    print("Training model for visualization...")
    model = DiagnosticModel()
    data = model.load_real_data("data")
    if data.empty:
        data = model.simulate_data(n_samples=1000)
    model.train(data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Pathology Distribution
    ax1 = plt.subplot(2, 3, 1)
    pathology_counts = data['PATHOLOGY'].value_counts()
    colors = sns.color_palette("husl", len(pathology_counts))
    pathology_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Distribution of Pathologies in Training Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pathology', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Age Distribution by Pathology
    ax2 = plt.subplot(2, 3, 2)
    for pathology in data['PATHOLOGY'].unique()[:5]:  # Top 5 pathologies
        subset = data[data['PATHOLOGY'] == pathology]['AGE']
        ax2.hist(subset, alpha=0.5, label=pathology[:20], bins=20)
    ax2.set_title('Age Distribution by Pathology', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=8)
    
    # 3. Gender Distribution
    ax3 = plt.subplot(2, 3, 3)
    if 'SEX' in data.columns:
        sex_counts = data['SEX'].value_counts()
        colors_sex = ['#3498db', '#e74c3c']
        ax3.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
                colors=colors_sex, startangle=90)
        ax3.set_title('Gender Distribution in Dataset', fontsize=14, fontweight='bold')
    
    # 4. Model Prediction Confidence
    ax4 = plt.subplot(2, 3, 4)
    # Generate sample predictions
    test_samples = data.sample(min(100, len(data)))
    confidences = []
    for _, row in test_samples.iterrows():
        input_dict = {col: row[col] for col in model.feature_columns if col in row.index}
        try:
            predictions = model.predict_proba(input_dict, top_k=1)
            if predictions:
                confidences.append(predictions[0][1])
        except:
            continue
    
    if confidences:
        ax4.hist(confidences, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax4.set_title('Model Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Confidence Score', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend()
    
    # 5. Recommendation Scoring Breakdown
    ax5 = plt.subplot(2, 3, 5)
    recommender = RecommenderEngine()
    
    # Simulate different scenarios
    scenarios = {
        'Perfect Match\n(Same Gender,\nSame Zip,\nIn-Network)': 0.8,
        'Gender Mismatch\n(Diff Gender,\nSame Zip,\nIn-Network)': 0.6,
        'Far Distance\n(Same Gender,\nDiff Zip,\nIn-Network)': 0.53,
        'Out of Network\n(All Match\nbut Out-Network)': 0.0
    }
    
    bars = ax5.bar(range(len(scenarios)), scenarios.values(), 
                   color=['#27ae60', '#f39c12', '#e67e22', '#c0392b'])
    ax5.set_xticks(range(len(scenarios)))
    ax5.set_xticklabels(scenarios.keys(), fontsize=9)
    ax5.set_title('Provider Affinity Scores by Scenario', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Affinity Score', fontsize=11)
    ax5.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, scenarios.values())):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. System Architecture Flow
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create flow diagram
    flow_text = """
    NeuroTriage System Flow
    ═══════════════════════
    
    1. INPUT
       └─ Patient Symptoms (Age, Sex, Symptoms)
    
    2. DIAGNOSIS
       └─ XGBoost Model → Top 5 Diagnoses
    
    3. TAXONOMY MAPPING
       └─ Pathology → NUCC Taxonomy Code
    
    4. PROVIDER FETCH
       └─ NPPES API (by Taxonomy + Zip)
    
    5. INSURANCE FILTER
       └─ Stream MRF JSON (ijson)
       └─ Filter by In-Network Status
    
    6. RANKING
       └─ Habit Match Algorithm
       └─ Score = Gender(0.2) + Distance(0.3) + Network(0.5)
    
    7. OUTPUT
       └─ Ranked List of Providers
    """
    
    ax6.text(0.1, 0.95, flow_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'neurotriage_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Additional: Create a simple metrics summary
    print("\n" + "="*60)
    print("NEUROTRIAGE SYSTEM METRICS")
    print("="*60)
    print(f"Total Training Samples: {len(data)}")
    print(f"Unique Pathologies: {data['PATHOLOGY'].nunique()}")
    print(f"Average Prediction Confidence: {np.mean(confidences):.3f}" if confidences else "N/A")
    print(f"Model Features: {len(model.feature_columns)}")
    print("="*60)

if __name__ == "__main__":
    visualize_system()
