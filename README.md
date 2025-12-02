# 🤖 Sentiment Analyzer with Machine Learning

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20Numpy%20%7C%20Matplotlib-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive project for sentiment classification (Positive/Negative) of text-based comments. This system implements, trains, and compares the performance of four Machine Learning algorithms: **Multi-layer Perceptron (MLP)**, **Support Vector Machine (SVM)**, **Random Forest**, and **Naive Bayes**.

The project includes a graphical user interface (GUI) developed with Tkinter for interactive testing, along with scripts for performance analysis and comparative chart generation.

---

## ✨ Features

- **4-Model Comparison:** Performance evaluation across MLP, SVM, Random Forest, and Naive Bayes.
- **Graphical User Interface (GUI):** A desktop application to classify new comments in real-time.
- **Performance Analysis:** Scripts to generate detailed metrics (Accuracy, Precision, Recall, F1-Score) and execution times.
- **Chart Generation:** Automatically creates 6 professional-grade visualizations for results analysis.
- **Modular Structure:** The code is organized into components, making it easy to maintain and add new models.

---

## 📊 Model Performance

The comparative analysis demonstrated the superiority of the MLP model, which achieved the best balance between accuracy and prediction speed.

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Prediction Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | **84.71%** | 84% | 85% | 0.85 | 2.13s | 0.02s |
| **SVM** | **83.08%** | 83% | 83% | 0.82 | 12.84s | 1.44s |
| **Random Forest** | **82.66%** | 82% | 83% | 0.82 | 8.21s | 0.07s |
| **Naive Bayes** | 67.08% | 45% | 67% | 0.54 | 0.01s | 0.02s |

---

## 🖼️ Graphical User Interface

The application allows for interactive testing of the models.

*(You can add a screenshot of your `classifier_app.py` application in action here)*
![GUI Screenshot](https://i.imgur.com/your_image_link.png)

---

## 🚀 How to Run the Project

Follow the steps below to set up and run the project locally.

### Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)

### 1. Clone the Repository

```bash
git clone https://github.com/victor-alca/IA_1_RP.git
cd IA_1_RP
```

### 2. Create and Activate a Virtual Environment

This creates an isolated environment to install the project's dependencies.

**On Windows (PowerShell):**
```powershell
# Create the environment
python -m venv venv

# Activate the environment
.\venv\Scripts\Activate.ps1
```

**On Linux or macOS:**
```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```
> You'll know the environment is active when `(venv)` appears at the beginning of your terminal prompt.

### 3. Install Dependencies

With the environment activated, install all required libraries.

```bash
pip install -r requirements.txt
```

### 4. Run the Scripts

You can now use the project's features.

**A) Launch the GUI:**
```bash
python src/classifier_app.py
```

**B) Compare Model Performance:**
This script trains all 4 models and displays the results table in the terminal.
```bash
python src/compare_models.py
```

**C) Generate Result Charts:**
This script creates 6 `.png` files with comparative charts in the `src/` folder.
```bash
python src/generate_plots.py
```

---

## 📂 Project Structure

```
.
├── data/
│   ├── PALAVRASpc.txt      # Vocabulary file used for text vectorization.
│   ├── CLtx.dat            # Contains the raw text data and their corresponding class labels.
│   ├── WTEXpc.dat          # Contains the vectorized text data (features).
│   └── WWRDpc.dat          # Contains word-related matrices for the vectorization process.
├── src/
│   ├── pkl/                # Stores pre-trained (serialized) models.
│   ├── classifier_app.py   # Logic and components for the GUI.
│   ├── comment_classifier.py # Main class that encapsulates the model.
│   ├── compare_models.py   # Script to compare the 4 models.
│   ├── data_loader.py      # Loads and preprocesses the data.
│   ├── generate_plots.py   # Script to create the result charts.
│   ├── gui_components.py   # Helper functions for the GUI.
│   ├── main.py             # Entry point to start the GUI application.
│   ├── model.py            # Implementation of the 4 ML models.
│   └── text_vectorizer.py  # Converts text into numerical vectors.
├── tests/                  # Unit tests (to be implemented).
├── requirements.txt        # List of project dependencies.
└── README.md               # This file.
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
