import tkinter as tk
from tkinter import messagebox, ttk
from comment_classifier import CommentClassifier
from gui_components import GUIComponents

class CommentClassifierApp:
    """
    This module defines the main application class for the Comment Classification GUI.
    It handles the initialization of the interface, user interactions, and integrates the comment classifier logic.
    Supports multiple ML models: MLP, Naive Bayes, Random Forest, and SVM
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Comentários - Reconhecimento de Padrões")
        self.root.geometry("800x750")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.classifier = None
        self.current_model_type = 'mlp'  # Default model
        
        # GUI components
        self.text_input = None
        self.classify_btn = None
        self.classification_label = None
        self.confidence_label = None
        self.coverage_label = None
        self.model_label = None
        self.progress = None
        self.status_bar = None
        self.model_combo = None
        
        # Create GUI first
        self.create_gui()
        
        # Load default classifier
        self.load_classifier()
        self.update_status("Pronto para classificar comentários!")
    
    def load_classifier(self):
        """Load or train the classifier with selected model"""
        try:
            self.update_status(f"Carregando modelo {self.current_model_type}...")
            self.classifier = CommentClassifier(model_type=self.current_model_type)
            self.classifier.load_or_train_model()
            
            # Update model label
            if hasattr(self.classifier.model, 'model_name'):
                self.model_label.config(text=f"Modelo atual: {self.classifier.model.model_name}")
            
            self.update_status(f"Modelo {self.current_model_type} carregado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
            self.update_status("Erro ao carregar modelo")
    
    def create_gui(self):
        """Create the graphical user interface using GUIComponents"""
        # Title section
        GUIComponents.create_title_section(self.root)
        
        # Model selection section
        model_frame = tk.Frame(self.root, bg='#f0f0f0')
        model_frame.pack(pady=10, padx=20, fill='x')
        
        model_select_label = tk.Label(
            model_frame, 
            text="Selecione o Modelo:", 
            font=('Arial', 11, 'bold'),
            bg='#f0f0f0'
        )
        model_select_label.pack(side='left', padx=5)
        
        # Model dropdown
        model_options = [
            'mlp - Multi-layer Perceptron',
            'naive_bayes - Naive Bayes',
            'random_forest - Random Forest',
            'svm - Support Vector Machine'
        ]
        
        self.model_combo = ttk.Combobox(
            model_frame,
            values=model_options,
            state='readonly',
            width=30,
            font=('Arial', 10)
        )
        self.model_combo.set(model_options[0])
        self.model_combo.pack(side='left', padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.change_model)
        
        # Current model label
        self.model_label = tk.Label(
            model_frame,
            text="Modelo atual: Multi-layer Perceptron",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#27ae60'
        )
        self.model_label.pack(side='left', padx=10)
        
        # Input section
        input_frame, self.text_input, button_frame = GUIComponents.create_input_section(self.root)
        
        # Buttons
        self.classify_btn = GUIComponents.create_classify_button(button_frame, self.classify_text)
        self.classify_btn.pack(side='left', padx=5)
        
        clear_btn = GUIComponents.create_clear_button(button_frame, self.clear_text)
        clear_btn.pack(side='left', padx=5)
        
        # Results section
        results_frame, self.classification_label, self.confidence_label, \
        self.coverage_label, self.progress = GUIComponents.create_results_section(self.root)
        
        # Status bar
        self.status_bar = GUIComponents.create_status_bar(self.root)
        self.status_bar.pack(side='bottom', fill='x')
    
    def change_model(self, event=None):
        """Change the classification model"""
        selected = self.model_combo.get()
        model_type = selected.split(' - ')[0]
        
        if model_type != self.current_model_type:
            self.current_model_type = model_type
            self.clear_results()
            self.load_classifier()
    
    def classify_text(self):
        """Classify the input text using CommentClassifier"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Aviso", "Por favor, digite um texto para classificar.")
            return
        
        try:
            # Show progress
            self.progress.start()
            self.classify_btn.config(state='disabled')
            self.update_status("Classificando comentário...")
            self.root.update()
            
            # Classify using the classifier
            result = self.classifier.classify_comment(text)
            
            # Extract results
            prediction = result['prediction']
            probabilities = result['probabilities']
            coverage = result['coverage']
            found_words = result['found_words']
            total_words = result['total_words']
            
            # Display results
            classification = "😊 POSITIVO" if prediction == 1 else "😞 NEGATIVO"
            color = "#27ae60" if prediction == 1 else "#e74c3c"
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            self.classification_label.config(text=classification, fg=color)
            self.confidence_label.config(text=f"Confiança: {confidence:.1%}")
            self.coverage_label.config(text=f"Palavras encontradas: {coverage:.1%} ({found_words}/{total_words} palavras)")
            
            self.update_status("Classificação concluída!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na classificação: {str(e)}")
            self.update_status("Erro na classificação")
        finally:
            # Hide progress
            self.progress.stop()
            self.classify_btn.config(state='normal')
    
    def clear_text(self):
        """Clear the input text and results"""
        self.text_input.delete("1.0", tk.END)
        self.clear_results()
        self.update_status("Texto limpo")
    
    def clear_results(self):
        """Clear only the classification results"""
        self.classification_label.config(text="Aguardando classificação...", fg='#7f8c8d')
        self.confidence_label.config(text="")
        self.coverage_label.config(text="")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    CommentClassifierApp(root)
    
    # Center window on screen
    root.eval('tk::PlaceWindow . center')
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()