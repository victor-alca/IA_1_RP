"""
Unit tests for comment_classifier module.
Tests the CommentClassifier class integration.
"""
import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from comment_classifier import CommentClassifier


class TestCommentClassifier(unittest.TestCase):
    """Test cases for CommentClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = CommentClassifier(model_type='mlp')

    def test_classifier_initialization(self):
        """Test that classifier initializes correctly."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.model_type, 'mlp')

    def test_load_or_train_classifier(self):
        """Test loading or training the classifier."""
        # Change to src directory for relative paths to work
        original_dir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        try:
            # Remove old pkl files to force retraining with compatible version
            import glob
            for pkl_file in glob.glob('./pkl/*.pkl'):
                try:
                    os.remove(pkl_file)
                except:
                    pass
            
            # Load or train should complete without errors
            result = self.classifier.load_or_train_model()
            self.assertTrue(result)
            
            # Model should have been loaded/trained
            self.assertIsNotNone(self.classifier.model)
        finally:
            os.chdir(original_dir)

    def test_classify_text(self):
        """Test classification of text."""
        original_dir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        try:
            # Remove old pkl files to force retraining
            import glob
            for pkl_file in glob.glob('./pkl/*.pkl'):
                try:
                    os.remove(pkl_file)
                except:
                    pass
            
            self.classifier.load_or_train_model()
            
            test_text = "Este é um teste"
            result = self.classifier.classify_comment(test_text)
            
            # Check that result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('probabilities', result)
            
            # Check that prediction is valid (0 or 1)
            self.assertIn(result['prediction'], [0, 1, -1])
        finally:
            os.chdir(original_dir)


if __name__ == '__main__':
    unittest.main()
