from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class MLPModel:
    """
    Class to train and evaluate a Multi-layer Perceptron (MLP) model for classification tasks.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = MLPClassifier()

    def split_data(self, X, y, test_size=0.3):
        """
        Splits the data for training and testing.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Trains the MLP model on the training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test data and prints metrics.
        """
        y_pred = self.model.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
