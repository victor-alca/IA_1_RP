import numpy as np

class TextVectorizer:
    """
    Utility for vectorizing new texts using PALAVRASpc.txt and WWRDpc.dat.
    Each word in the vocabulary is mapped to a vector. A text is represented
    by the mean of the vectors of the words it contains (words not in the vocabulary are ignored).
    """
    def __init__(self, vocab_path, vectors_path):
        # Load vocabulary and word vectors
        self.vocab = self._load_vocab(vocab_path)
        self.word_vectors = self._load_vectors(vectors_path)
        self.word_to_vec = self._build_word_to_vec()

    def _load_vocab(self, path):
        # Load vocabulary (one word per line)
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def _load_vectors(self, path):
        # Load word vectors (one vector per line)
        return np.loadtxt(path)

    def _build_word_to_vec(self):
        # Create a dictionary mapping words to their vectors
        return {word: vec for word, vec in zip(self.vocab, self.word_vectors)}

    def vectorize(self, text):
        """
        Convert a text (string) into a vector by averaging the vectors of the words it contains.
        Words not in the vocabulary are ignored. If no words are found, returns a zero vector.
        """
        # Convert to uppercase to match vocabulary format
        words = text.strip().upper().split()
        vectors = [self.word_to_vec[w] for w in words if w in self.word_to_vec]
        if not vectors:
            return np.zeros(self.word_vectors.shape[1])
        return np.mean(vectors, axis=0)
