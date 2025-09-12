import numpy as np

class DataLoader:
    """
    Class to load and prepare data from .dat files.
    """
    def __init__(self, features_path, labels_path):
        """
        Initializes the DataLoader with paths to features and labels files.
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.X = None
        self.y = None

    def load_data(self):
        """
        Loads the data from the files and stores it in self.X and self.y.
        """
        self.X = np.loadtxt(self.features_path)
        self.y = np.loadtxt(self.labels_path)
        return self.X, self.y
