from data_loader import DataLoader
from model import MLPModel

FEATURES_PATH = '../data/WTEXpc.dat'
LABELS_PATH = '../data/CLtx.dat'

if __name__ == "__main__":
    # Load data
    loader = DataLoader(FEATURES_PATH, LABELS_PATH)
    X, y = loader.load_data()

    # Initialize model
    mlp = MLPModel()

    # Split data
    X_train, X_test, y_train, y_test = mlp.split_data(X, y)

    # Train model
    mlp.train(X_train, y_train)

    # Evaluate model
    mlp.evaluate(X_test, y_test)
