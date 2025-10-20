# Utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# def print_model_scores(scores_dict):
#     print("Scores des modèles :")
#     print("{:<20} {:<10} {:<10} {:<10}".format("Modèle", "Accuracy", "Precision", "Recall"))
#     print("-"*55)
#     for model, metrics in scores_dict.items():
#         print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
#             model, metrics['accuracy'], metrics['precision'], metrics['recall']
#         ))


def print_model_scores(scores: Dict[str, Dict[str, float]]):
    """Display model performance metrics in formatted table"""
    df_scores = pd.DataFrame(scores).T * 100
    df_scores = df_scores.round(2)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(df_scores.to_string())
    
    best_model = df_scores['accuracy'].idxmax()
    best_accuracy = df_scores.loc[best_model, 'accuracy']
    print(f"\nBEST MODEL: {best_model} ({best_accuracy}% accuracy)")
    
    return df_scores

def plot_model_comparison(scores: Dict[str, Dict[str, float]]):
    """Generate performance comparison visualizations"""
    df_scores = pd.DataFrame(scores).T * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Metrics Comparison')
    
    metrics = ['accuracy', 'precision', 'recall']
    for i, metric in enumerate(metrics):
        model_names = list(scores.keys())
        values = [scores[model][metric] * 100 for model in model_names]
        
        bars = axes[i].bar(model_names, values, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel(f'{metric.capitalize()} (%)')
        axes[i].set_ylim(0, 100)
        
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def print_performance_analysis(scores: Dict[str, Dict[str, float]]):
    """Detailed performance analysis with categorical assessment"""
    print("\nPERFORMANCE ANALYSIS")
    print("-" * 30)
    
    df_scores = pd.DataFrame(scores).T * 100
    
    for model in scores.keys():
        acc = df_scores.loc[model, 'accuracy']
        prec = df_scores.loc[model, 'precision']
        rec = df_scores.loc[model, 'recall']
        
        print(f"\n{model}:")
        print(f"  Accuracy:  {acc:6.2f}%")
        print(f"  Precision: {prec:6.2f}%") 
        print(f"  Recall:    {rec:6.2f}%")
        
        # F1 score calculation
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  F1 Score:  {f1:6.2f}%")
    
    best_model = df_scores['accuracy'].idxmax()
    print(f"\nRECOMMENDED MODEL: {best_model}")

