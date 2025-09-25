import pickle
import os
from data_loader import DataLoader
from model import MLPModel
from text_vectorizer import TextVectorizer


class CommentClassifier:
    """
    Comment Classification Model Handler
    Handles model loading, training, and prediction operations
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
        # Data file paths
        self.FEATURES_PATH = '../data/WTEXpc.dat'
        self.LABELS_PATH = '../data/CLtx.dat'
        self.VOCAB_PATH = '../data/PALAVRASpc.txt'
        self.WORD_VECTORS_PATH = '../data/WWRDpc.dat'

        PKL_DIR = os.path.join(os.path.dirname(__file__), "pkl")
        os.makedirs(PKL_DIR, exist_ok=True)
        
        # Model save paths
        self.MODEL_PATH = './pkl/comment_model.pkl'
        self.VECTORIZER_PATH = './pkl/vectorizer.pkl'
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if self._model_exists():
                self._load_saved_model()
            else:
                self._train_new_model()
            return True
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _model_exists(self):
        """Check if saved model files exist"""
        return os.path.exists(self.MODEL_PATH) and os.path.exists(self.VECTORIZER_PATH)
    
    def _load_saved_model(self):
        """Load pre-trained model from pickle files"""
        with open(self.MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def _train_new_model(self):
        """Train a new model and save it"""
        # Load data and train model
        loader = DataLoader(self.FEATURES_PATH, self.LABELS_PATH)
        X, y = loader.load_data()
        
        # Initialize and train model
        self.model = MLPModel()
        X_train, X_test, y_train, y_test = self.model.split_data(X, y)
        self.model.train(X_train, y_train)
        
        # Initialize vectorizer
        self.vectorizer = TextVectorizer(self.VOCAB_PATH, self.WORD_VECTORS_PATH)
        
        # Save model and vectorizer
        self._save_model()
    
    def _save_model(self):
        """Save trained model and vectorizer"""
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def classify_comment(self, text):
        """
        Classify a comment text
        Returns: dict with prediction, probabilities, and coverage info
        """
        if not self.model or not self.vectorizer:
            raise Exception("Model not loaded. Call load_or_train_model() first.")
        
        # Vectorize text
        text_vector = self.vectorizer.vectorize(text)
        
        # Make prediction
        prediction = self.model.model.predict([text_vector])[0]
        probabilities = self.model.model.predict_proba([text_vector])[0]
        
        # Calculate vocabulary coverage
        text_processed = self.vectorizer._remove_accents(text)
        words = text_processed.upper().split()
        found_words = [w for w in words if w in self.vectorizer.word_to_vec]
        coverage = len(found_words) / len(words) if words else 0
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'coverage': coverage,
            'found_words': len(found_words),
            'total_words': len(words)
        }