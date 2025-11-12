from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class ClassificationModel:
    """
    Generic class to train and evaluate different classification models.
    Supports: MLP, Naive Bayes, Random Forest, and SVM
    """
    
    AVAILABLE_MODELS = {
        'mlp': ('Multi-layer Perceptron', MLPClassifier),
        'naive_bayes': ('Naive Bayes', MultinomialNB),
        'random_forest': ('Random Forest', RandomForestClassifier),
        'svm': ('Support Vector Machine', SVC)
    }
    
    def __init__(self, model_type='mlp', random_state=42):
        """
        Initialize the classification model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options: 'mlp', 'naive_bayes', 'random_forest', 'svm'
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model_type = model_type
        
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name, model_class = self.AVAILABLE_MODELS[model_type]
        self.model = self._initialize_model(model_class)
    
    def _initialize_model(self, model_class):
        """Initialize the specific model with optimized hyperparameters"""
        if self.model_type == 'mlp':
            # Multi-layer Perceptron with 2 hidden layers (100, 50 neurons)
            # Uses adaptive learning rate and early stopping for optimal convergence
            return model_class(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=self.random_state,
                verbose=False
            )
        elif self.model_type == 'naive_bayes':
            # Multinomial Naive Bayes with Laplace smoothing
            # Simple probabilistic classifier based on Bayes' theorem
            return model_class(alpha=1.0, fit_prior=True)
        elif self.model_type == 'random_forest':
            # Random Forest ensemble with 200 trees and balanced class weights
            # Uses parallel processing (n_jobs=-1) for faster training
            return model_class(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap = True,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
        elif self.model_type == 'svm':
            # Support Vector Machine with RBF kernel for non-linear classification
            # Optimized for accuracy without class weight balancing
            return model_class(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                cache_size=500,
                max_iter=-1,
                random_state=self.random_state,
                verbose=False
            )
        return model_class()

    def split_data(self, X, y, test_size=0.3):
        """
        Splits the data for training and testing.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Trains the model on the training data.
        """
        # For Naive Bayes, ensure non-negative values
        if self.model_type == 'naive_bayes':
            X_train = np.abs(X_train)
        
        self.model.fit(X_train, y_train)
        print(f"\n{self.model_name} trained successfully!")

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test data and prints metrics.
        Returns accuracy, classification report, and confusion matrix.
        """
        # For Naive Bayes, ensure non-negative values
        if self.model_type == 'naive_bayes':
            X_test = np.abs(X_test)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        print(f"Confusion Matrix:")
        print(conf_matrix)
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'model_name': self.model_name
        }
