import numpy as np

# Importa o módulo para carregar os dados
from sklearn.model_selection import train_test_split

# Importa um modelo de classificação, como a Support Vector Machine
from sklearn.neural_network import MLPClassifier

# Importa as métricas para avaliar o modelo
from sklearn.metrics import accuracy_score, classification_report

"""
A pasta DADOS contém os seguintes arquivos:


1) PALAVRASpc.txt- Lista de palavras vetorizadas, uma por linha (n= 9538).

2) WWRDpc.dat- Vetores com 100 coordenadas, um por linha, correspondentes às palavras do item anterior (n=9538) ;

3) WTEXpc.dat- Vetores com 100 coordenadas de textos classificados (n=10400). Cada vetor de texto é calculado como a média dos vetores das palavras que ocorrem nele, sendo que as palavras que não estão na lista apresentada devem ser ignoradas.

4) CLtx.dat- Classificação de cada texto de WTEXpc. Textos considerados como positivos são classificados com 1 (um) na linha correspondente de CLtx e 0 (zero) caso a supervisão indique como sendo um texto negativo.

O Trabalho


A proposta do trabalho é desenvolver um modelo de reconhecimento de padrões para classificar comentários, com base nos dados fornecidos. O roteiro sugerido consiste em:


a) Treinar e testar um modelo de classificação baseado nos vetores do item 3 com as classes do item 4.

b) Determinar um novo conjunto de textos (positivos e negativos) e verificar a habilidade de classificação do modelo em casos reais. Discutir resultados.

c) Propor uma ferramenta de classificação executável com entrada livre para o usuário (uma-a-uma).
"""

# X = Features (matriz dos dados de entrada)
# y = label (rótulo/classes dos valores para prever as amostras de X)

X = np.loadtxt('../data/WTEXpc.dat')
y = np.loadtxt('../data/CLtx.dat')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))    