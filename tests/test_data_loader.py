"""
Unit tests for data_loader module.
Tests data loading and preprocessing functionality.
"""
import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use actual data paths relative to tests directory
        base_path = os.path.join(os.path.dirname(__file__), '..')
        self.features_path = os.path.join(base_path, 'data', 'WTEXpc.dat')
        self.labels_path = os.path.join(base_path, 'data', 'CLtx.dat')

    def test_dataloader_initialization(self):
        """Test that DataLoader initializes correctly."""
        loader = DataLoader(self.features_path, self.labels_path)
        self.assertIsNotNone(loader)
        self.assertEqual(loader.features_path, self.features_path)
        self.assertEqual(loader.labels_path, self.labels_path)

    def test_load_data_returns_correct_types(self):
        """Test that load_data returns numpy arrays."""
        loader = DataLoader(self.features_path, self.labels_path)
        X, y = loader.load_data()
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_load_data_correct_shapes(self):
        """Test that data has correct dimensions."""
        loader = DataLoader(self.features_path, self.labels_path)
        X, y = loader.load_data()
        
        # Check that X has 100 features
        self.assertEqual(X.shape[1], 100)
        
        # Check that labels match data length
        self.assertEqual(len(y), len(X))

    def test_labels_are_binary(self):
        """Test that labels contain only 0 and 1."""
        loader = DataLoader(self.features_path, self.labels_path)
        X, y = loader.load_data()
        
        # Check labels are exactly 0 and 1
        unique_labels = np.unique(y)
        self.assertEqual(len(unique_labels), 2)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))

    def test_data_not_empty(self):
        """Test that loaded data is not empty."""
        loader = DataLoader(self.features_path, self.labels_path)
        X, y = loader.load_data()
        
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)


if __name__ == '__main__':
    unittest.main()
