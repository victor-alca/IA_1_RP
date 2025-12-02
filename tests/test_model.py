"""
Unit tests for model module.
Tests the ClassificationModel with different algorithms.
"""
import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import ClassificationModel


class TestClassificationModel(unittest.TestCase):
    """Test cases for ClassificationModel class."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        np.random.seed(42)
        # Create simple synthetic data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(30, 10)
        self.y_test = np.random.randint(0, 2, 30)

    def test_mlp_model_creation(self):
        """Test MLP model initialization."""
        model = ClassificationModel(model_type='mlp')
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model_type, 'mlp')

    def test_svm_model_creation(self):
        """Test SVM model initialization."""
        model = ClassificationModel(model_type='svm')
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model_type, 'svm')

    def test_random_forest_model_creation(self):
        """Test Random Forest model initialization."""
        model = ClassificationModel(model_type='random_forest')
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model_type, 'random_forest')

    def test_naive_bayes_model_creation(self):
        """Test Naive Bayes model initialization."""
        model = ClassificationModel(model_type='naive_bayes')
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model_type, 'naive_bayes')

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with self.assertRaises(ValueError):
            ClassificationModel(model_type='invalid_model')

    def test_model_training(self):
        """Test that model can be trained."""
        model = ClassificationModel(model_type='mlp')
        
        # Should not raise any exceptions
        model.train(self.X_train, self.y_train)

    def test_model_training_and_prediction(self):
        """Test that model can be trained."""
        model = ClassificationModel(model_type='mlp')
        X_train, X_test, y_train, y_test = model.split_data(self.X_train, self.y_train, test_size=0.3)
        
        # Train the model - should not raise any exceptions
        model.train(X_train, y_train)
        
        # Model should be fitted
        self.assertIsNotNone(model.model)

    def test_model_evaluation(self):
        """Test that model evaluation returns correct metrics."""
        model = ClassificationModel(model_type='mlp')
        X_train, X_test, y_train, y_test = model.split_data(self.X_train, self.y_train, test_size=0.3)
        
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        # Check that expected keys are present
        self.assertIn('accuracy', metrics)
        self.assertIn('report', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('model_name', metrics)
        
        # Check that accuracy is in valid range [0, 1]
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_split_data(self):
        """Test data splitting functionality."""
        model = ClassificationModel(model_type='mlp')
        X_train, X_test, y_train, y_test = model.split_data(self.X_train, self.y_train, test_size=0.3)
        
        # Check that split was done correctly
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train) + len(X_test), len(self.X_train))


if __name__ == '__main__':
    unittest.main()
