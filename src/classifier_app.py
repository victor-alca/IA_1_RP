import tkinter as tk
from tkinter import messagebox
from comment_classifier import CommentClassifier
from gui_components import GUIComponents

class CommentClassifierApp:
    """
    This module defines the main application class for the Comment Classification GUI.
    It handles the initialization of the interface, user interactions, and integrates the comment classifier logic.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Comentários - Reconhecimento de Padrões")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.classifier = CommentClassifier()
        
        # GUI components
        self.text_input = None
        self.classify_btn = None
        self.classification_label = None
        self.confidence_label = None
        self.coverage_label = None
        self.progress = None
        self.status_bar = None
        
        # Load model and create GUI
        self.load_classifier()
        self.create_gui()
        self.update_status("Pronto para classificar comentários!")
    
    def load_classifier(self):
        """Load or train the classifier"""
        try:
            self.classifier.load_or_train_model()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
    
    def create_gui(self):
        """Create the graphical user interface using GUIComponents"""
        # Title section
        GUIComponents.create_title_section(self.root)
        
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
        """Clear the input text"""
        self.text_input.delete("1.0", tk.END)
        self.classification_label.config(text="Aguardando classificação...", fg='#7f8c8d')
        self.confidence_label.config(text="")
        self.coverage_label.config(text="")
        self.update_status("Texto limpo")
    
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