"""
Unit tests for text_vectorizer module.
Tests text preprocessing and vectorization functionality.
"""
import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_vectorizer import TextVectorizer


class TestTextVectorizer(unittest.TestCase):
    """Test cases for text vectorization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        vocab_path = os.path.join(base_path, 'data', 'PALAVRASpc.txt')
        vectors_path = os.path.join(base_path, 'data', 'WWRDpc.dat')
        self.vectorizer = TextVectorizer(vocab_path, vectors_path)

    def test_vectorizer_initialization(self):
        """Test that vectorizer initializes correctly."""
        self.assertIsNotNone(self.vectorizer.vocab)
        self.assertIsNotNone(self.vectorizer.word_vectors)
        self.assertGreater(len(self.vectorizer.vocab), 0)

    def test_remove_accents(self):
        """Test accent removal functionality."""
        text = "José não está aqui. Ação, coração!"
        expected = "Jose nao esta aqui. Acao, coracao!"
        result = self.vectorizer._remove_accents(text)
        self.assertEqual(result, expected)

    def test_vectorize_returns_correct_shape(self):
        """Test that vectorization returns correct shape."""
        text = "Este é um comentário de teste"
        vector = self.vectorizer.vectorize(text)
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (100,))

    def test_vectorize_empty_string(self):
        """Test vectorization of empty string."""
        vector = self.vectorizer.vectorize("")
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (100,))

    def test_vectorize_unknown_words(self):
        """Test vectorization of text with unknown words."""
        text = "xyzabc123 palavraqueprovavelmentenaoexiste"
        vector = self.vectorizer.vectorize(text)
        
        # Should still return a valid vector
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (100,))

    def test_consistent_vectorization(self):
        """Test that same text produces same vector."""
        text = "comentário positivo"
        
        vector1 = self.vectorizer.vectorize(text)
        vector2 = self.vectorizer.vectorize(text)
        
        np.testing.assert_array_equal(vector1, vector2)


if __name__ == '__main__':
    unittest.main()
