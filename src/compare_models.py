"""
Model Comparison Script
Trains and compares different classification models for comment classification
"""

from data_loader import DataLoader
from model import ClassificationModel
import time


FEATURES_PATH = '../data/WTEXpc.dat'
LABELS_PATH = '../data/CLtx.dat'

def compare_all_models():
    """
    Train and compare all available models
    """
    # Load data once
    print("Loading data...")
    loader = DataLoader(FEATURES_PATH, LABELS_PATH)
    X, y = loader.load_data()
    
    print(f"\nDataset Info:")
    print(f"Total samples: {len(y)}")
    print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Negative samples: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print(f"Features dimension: {X.shape[1]}")
    
    # Models to compare
    models_to_test = ['mlp', 'naive_bayes', 'random_forest', 'svm']
    
    results = []
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING ALL MODELS")
    print("="*70)
    
    for model_type in models_to_test:
        print(f"\n{'*'*70}")
        print(f"Testing model: {model_type.upper()}")
        print(f"{'*'*70}")
        
        try:
            # Initialize model
            model = ClassificationModel(model_type=model_type, random_state=42)
            
            # Split data
            X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.3)
            
            # Train and measure time
            start_time = time.time()
            model.train(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            metrics = model.evaluate(X_test, y_test)
            prediction_time = time.time() - start_time
            
            # Store results
            results.append({
                'model_type': model_type,
                'model_name': metrics['model_name'],
                'accuracy': metrics['accuracy'],
                'training_time': training_time,
                'prediction_time': prediction_time,
                'confusion_matrix': metrics['confusion_matrix']
            })
            
            print(f"Training time: {training_time:.4f}s")
            print(f"Prediction time: {prediction_time:.4f}s")
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    # Print comparison summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY - MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Train Time':<15} {'Pred Time':<15}")
    print("-"*70)
    
    for result in results:
        print(f"{result['model_name']:<25} "
              f"{result['accuracy']:<12.4f} "
              f"{result['training_time']:<15.4f}s "
              f"{result['prediction_time']:<15.4f}s")
    
    print("="*70)
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest Model: {best_model['model_name']} "
              f"with accuracy of {best_model['accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    results = compare_all_models()
