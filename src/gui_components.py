

import tkinter as tk
from tkinter import ttk, scrolledtext


class GUIComponents:
    """
    Factory class for creating GUI components
    Creates and manages GUI components for the comment classifier
    """
    
    @staticmethod
    def create_title_section(parent):
        """Create the title section"""
        title_frame = tk.Frame(parent, bg='#f0f0f0')
        title_frame.pack(pady=10, fill='x')
        
        title_label = tk.Label(title_frame, 
                              text="🤖 Classificador de Comentários", 
                              font=('Arial', 18, 'bold'),
                              bg='#f0f0f0', fg='#2c3e50')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, 
                                 text="Classificação de comentários em tempo real",
                                 font=('Arial', 10),
                                 bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack()
        
        return title_frame
    
    @staticmethod
    def create_input_section(parent):
        """Create the input section with text area and buttons"""
        input_frame = tk.LabelFrame(parent, text="📝 Digite seu texto", 
                                   font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', fg='#2c3e50')
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Text input area
        text_input = scrolledtext.ScrolledText(input_frame, 
                                              height=8, 
                                              font=('Arial', 11),
                                              wrap=tk.WORD)
        text_input.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(pady=5, fill='x')
        
        return input_frame, text_input, button_frame
    
    @staticmethod
    def create_classify_button(parent, command):
        """Create classify button"""
        return tk.Button(parent, 
                        text="🔍 Classificar",
                        command=command,
                        bg='#3498db', fg='white',
                        font=('Arial', 12, 'bold'),
                        relief='flat',
                        padx=20, pady=5)
    
    @staticmethod
    def create_clear_button(parent, command):
        """Create clear button"""
        return tk.Button(parent, 
                        text="🗑️ Limpar",
                        command=command,
                        bg='#95a5a6', fg='white',
                        font=('Arial', 12, 'bold'),
                        relief='flat',
                        padx=20, pady=5)
    
    @staticmethod
    def create_results_section(parent):
        """Create results display section"""
        results_frame = tk.LabelFrame(parent, text="📊 Resultado da Classificação", 
                                     font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0', fg='#2c3e50')
        results_frame.pack(pady=10, padx=20, fill='both')
        
        # Result display frame
        result_frame = tk.Frame(results_frame, bg='#f0f0f0')
        result_frame.pack(pady=10, padx=10, fill='both')
        
        # Result labels
        classification_label = tk.Label(result_frame, 
                                       text="Aguardando classificação...",
                                       font=('Arial', 14, 'bold'),
                                       bg='#f0f0f0', fg='#7f8c8d')
        classification_label.pack(pady=5)
        
        confidence_label = tk.Label(result_frame, 
                                   text="",
                                   font=('Arial', 12),
                                   bg='#f0f0f0', fg='#7f8c8d')
        confidence_label.pack(pady=2)
        
        coverage_label = tk.Label(result_frame, 
                                 text="",
                                 font=('Arial', 10),
                                 bg='#f0f0f0', fg='#7f8c8d')
        coverage_label.pack(pady=2)
        
        # Progress bar
        progress = ttk.Progressbar(results_frame, mode='indeterminate')
        progress.pack(pady=5, padx=10, fill='x')
        
        return results_frame, classification_label, confidence_label, coverage_label, progress
    
    @staticmethod
    def create_status_bar(parent):
        """Create status bar"""
        return tk.Label(parent, 
                       text="Pronto",
                       relief='sunken',
                       anchor='w',
                       bg='#34495e', fg='white',
                       font=('Arial', 9))