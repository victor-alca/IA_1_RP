"""
Script to generate comparative visualizations of classification models
Useful for including in the report
"""

import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader
from model import ClassificationModel
import time


FEATURES_PATH = '../data/WTEXpc.dat'
LABELS_PATH = '../data/CLtx.dat'


def plot_model_comparison():
    """
    Generate comparative plots of models for the report
    """
    # Load data
    print("Loading data...")
    loader = DataLoader(FEATURES_PATH, LABELS_PATH)
    X, y = loader.load_data()
    
    # Models to compare
    models_to_test = ['mlp', 'naive_bayes', 'random_forest', 'svm']
    model_names = ['MLP', 'Naive Bayes', 'Random Forest', 'SVM']
    
    results = {
        'names': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'train_time': [],
        'pred_time': []
    }
    
    print("\nTraining models to generate plots...")
    
    for i, model_type in enumerate(models_to_test):
        print(f"Training {model_names[i]}...")
        
        try:
            # Initialize and train
            model = ClassificationModel(model_type=model_type, random_state=42)
            X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.3)
            
            start = time.time()
            model.train(X_train, y_train)
            train_time = time.time() - start
            
            # Get predictions
            if model_type == 'naive_bayes':
                X_test_eval = np.abs(X_test)
            else:
                X_test_eval = X_test
            
            start = time.time()
            y_pred = model.model.predict(X_test_eval)
            pred_time = time.time() - start
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results['names'].append(model_names[i])
            results['accuracy'].append(accuracy * 100)
            results['precision'].append(precision * 100)
            results['recall'].append(recall * 100)
            results['f1_score'].append(f1 * 100)
            results['train_time'].append(train_time)
            results['pred_time'].append(pred_time)
            
        except Exception as e:
            print(f"Error in {model_names[i]}: {str(e)}")
            continue
    
    # Create visualizations
    create_plots(results)
    
    print("\n Plots saved in 'src/' folder")
    print("   - 1_model_accuracy_comparison.png")
    print("   - 2_model_precision_recall.png")
    print("   - 3_model_f1_score.png")
    print("   - 4_model_training_time.png")
    print("   - 5_model_prediction_time.png")
    print("   - 6_model_metrics_radar.png")


def create_plots(results):
    """Create and save individual comparison plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # 1. Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['names'], results['accuracy'], color=colors, alpha=0.9)
    plt.ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12, fontweight='bold')
    plt.title('Comparação de Acurácia entre Modelos', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('1_model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 1 created: 1_model_accuracy_comparison.png")
    plt.close()
    
    # 2. Precision vs Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results['names']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, results['precision'], width, label='Precisão', 
                   color='#2ecc71', alpha=0.9)
    bars2 = ax.bar(x + width/2, results['recall'], width, label='Recall', 
                   color='#f39c12', alpha=0.9)
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_title('Precisão vs Recall por Modelo', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results['names'])
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('2_model_precision_recall.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 2 created: 2_model_precision_recall.png")
    plt.close()
    
    # 3. F1-Score Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['names'], results['f1_score'], color=colors, alpha=0.9)
    plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12, fontweight='bold')
    plt.title('Comparação de F1-Score entre Modelos', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('3_model_f1_score.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 3 created: 3_model_f1_score.png")
    plt.close()
    
    # 4. Training Time Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['names'], results['train_time'], color=colors, alpha=0.9)
    plt.ylabel('Tempo (segundos)', fontsize=12, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12, fontweight='bold')
    plt.title('Tempo de Treinamento por Modelo', fontsize=14, fontweight='bold', pad=20)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('4_model_training_time.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 4 created: 4_model_training_time.png")
    plt.close()
    
    # 5. Prediction Time Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['names'], results['pred_time'], color=colors, alpha=0.9)
    plt.ylabel('Tempo (segundos)', fontsize=12, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12, fontweight='bold')
    plt.title('Tempo de Predição por Modelo', fontsize=14, fontweight='bold', pad=20)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('5_model_prediction_time.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 5 created: 5_model_prediction_time.png")
    plt.close()
    
    # 6. Radar Chart - Multiple Metrics
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Normalize metrics to 0-100 scale
    categories = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Plot each model
    for i, model_name in enumerate(results['names']):
        values = [
            results['accuracy'][i],
            results['precision'][i],
            results['recall'][i],
            results['f1_score'][i]
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_title('Comparação Multidimensional de Métricas', 
                fontsize=14, fontweight='bold', pad=30, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('6_model_metrics_radar.png', dpi=300, bbox_inches='tight')
    print("✓ Plot 6 created: 6_model_metrics_radar.png")
    plt.close()


if __name__ == "__main__":
    plot_model_comparison()
