
from data_loader import DataLoader
from model import MLPModel
from text_vectorizer import TextVectorizer


FEATURES_PATH = '../data/WTEXpc.dat'
LABELS_PATH = '../data/CLtx.dat'
VOCAB_PATH = '../data/PALAVRASpc.txt'
WORD_VECTORS_PATH = '../data/WWRDpc.dat'

if __name__ == "__main__":
    # Load data
    loader = DataLoader(FEATURES_PATH, LABELS_PATH)
    X, y = loader.load_data()

    # Initialize model
    mlp = MLPModel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = mlp.split_data(X, y)

    # Train model
    mlp.train(X_train, y_train)

    # Evaluate model
    mlp.evaluate(X_test, y_test)
    
    # Check data distribution
    print(f"\nData distribution - Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
    print(f"Percentage positive: {sum(y)/len(y)*100:.1f}%")
    
    # Vectorize and classify multiple texts
    vectorizer = TextVectorizer(VOCAB_PATH, WORD_VECTORS_PATH)
    new_texts = [
        "produto excelente recomendo muito bom",
        "péssimo atendimento muito ruim não recomendo",
        "a semana passada, no dia 20 ou 21, fui até a loja Ella's, no Shopping Jardim das Américas, com meu marido e minha bebê para comprar alguns itens de maquiagem. Somos uma família negra e, infelizmente, passamos por uma situação extremamente desagradável e constrangedora.",
        "adorei o serviço rápido e eficiente",
        "entrega atrasada e produto defeituoso",
        "experiência incrível vale a pena",
        "não gostei nada da qualidade",
        "atendimento ao cliente excepcional",
        "preço justo e bom custo benefício",
        "ótimo produto com excelente qualidade",
        "serviço péssimo e atendimento horrível",
        "recomendação incrível vale muito à pena"
    ]
    
    print("\n--- Testing new texts ---")
    print("Sample vocabulary words:", list(vectorizer.word_to_vec.keys())[:10])
    
    for i, new_text in enumerate(new_texts, 1):
        text_vector = vectorizer.vectorize(new_text)
        # Predict the class (0=negative, 1=positive) using the trained MLP model
        pred = mlp.model.predict([text_vector])[0]
        # Get the prediction probabilities for both classes
        prob = mlp.model.predict_proba([text_vector])[0]
                
        # Check vocabulary coverage using processed text
        text_processed = vectorizer._remove_accents(new_text)
        words = text_processed.upper().split()
        found_words = [w for w in words if w in vectorizer.word_to_vec]
        
        print(f"\nTest {i}:")
        print(f"Original text: {new_text[:60]}{'...' if len(new_text) > 60 else ''}")
        print(f"Predicted class: {pred} (0=negative, 1=positive)")
        print(f"Confidence: negative={prob[0]:.3f}, positive={prob[1]:.3f}")
        print(f"Words in vocabulary: {len(found_words)}/{len(words)}")
        if found_words:
            print(f"Found words: {found_words}")
        else:
            print("No words found in vocabulary!")
            print(f"Looking for: {words}")
