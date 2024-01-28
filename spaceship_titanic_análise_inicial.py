#**Spaceship Titanic:**

#**Imports:**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

#**Carregando os Dados:**

dataset_titanic_train = pd.read_csv("train.csv")
dataset_titanic_test = pd.read_csv("test.csv")

#**Análise Exploratória de Dados Inicial:**

# Primeiras linhas
dataset_titanic_train.head()

# últimas linhas
dataset_titanic_train.tail()

# Formato da tabela
dataset_titanic_train.shape

# Info gerais
dataset_titanic_train.info

# Tipo dos dados
dataset_titanic_train.dtypes

# Resumo Estatístico
dataset_titanic_train.describe()

import missingno as msno
msno.matrix(dataset_titanic_train)

# Dataset de treino: Verificar os valores nulos em porcentagem
(dataset_titanic_train.isnull().sum()/dataset_titanic_train.shape[0]).sort_values(ascending = True)

import missingno as msno
msno.matrix(dataset_titanic_test)

# Dataset de teste: Verificar os valores nulos em porcentagem
(dataset_titanic_test.isnull().sum()/dataset_titanic_test.shape[0]).sort_values(ascending = True)

#**Tratamento de Dados Inicial:**

# O tratamento inicial focou em colunas numéricas, com o objetivo de analisar o comportamento do modelo com estes.**

#**1) Age: Verificação da distribuição dos dados:**

# Média de idade
dataset_titanic_train.Age.mean()

# Mediana da idade
dataset_titanic_train.Age.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["Age"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["Age"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.Age.isnull(),"Age"] = dataset_titanic_train["Age"].mean()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.Age.isnull(),"Age"] = dataset_titanic_test["Age"].mean()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

#**Substituição pela média em razão:**
# Da pouca diferença em relação a mediana
# Distribuição parcialmente normal

#**2) RoomService: Verificação da distribuição de dados**

# Média de gasto de serviço de quarto
dataset_titanic_train.RoomService.mean()

# Mediana de gasto de serviço de quarto
dataset_titanic_train.RoomService.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["RoomService"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["RoomService"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.RoomService.isnull(),"RoomService"] = dataset_titanic_train["RoomService"].median()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.RoomService.isnull(),"RoomService"] = dataset_titanic_test["RoomService"].median()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

# **Substituição pela mediana em razão:**
# Distribuição não normal
# Grande quantidade de outliers
# Mediana é pouco sensível aos outliers

#**3) FoodCourt: Verificação da distribuição de dados**

# Média de gasto em comida
dataset_titanic_train.FoodCourt.mean()

# Mediana de gasto em comida
dataset_titanic_train.FoodCourt.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["FoodCourt"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["FoodCourt"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.FoodCourt.isnull(),"FoodCourt"] = dataset_titanic_train["FoodCourt"].median()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.FoodCourt.isnull(),"FoodCourt"] = dataset_titanic_test["FoodCourt"].median()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

#**4) ShoppingMall: Verificação da distribuição de dados**

# Média de gasto de compras
dataset_titanic_train.ShoppingMall.mean()

# Mediana de gasto em compras
dataset_titanic_train.ShoppingMall.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["ShoppingMall"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["ShoppingMall"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.ShoppingMall.isnull(),"ShoppingMall"] = dataset_titanic_train["ShoppingMall"].median()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.ShoppingMall.isnull(),"ShoppingMall"] = dataset_titanic_test["ShoppingMall"].median()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

#**5) Spa: Verificação da distribuição de dados**

# Média de gasto em spa
dataset_titanic_train.Spa.mean()

# Mediana de gasto em spa
dataset_titanic_train.Spa.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["Spa"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["Spa"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.Spa.isnull(),"Spa"] = dataset_titanic_train["Spa"].median()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.Spa.isnull(),"Spa"] = dataset_titanic_test["Spa"].median()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

#**6) VRDeck: Verificação da distribuição de dados**

# Média de gasto em deck
dataset_titanic_train.VRDeck.mean()

# Mediana de gasto em deck
dataset_titanic_train.VRDeck.median()

# Verificar distribuição com KDEPlot
sea.kdeplot(dataset_titanic_train["VRDeck"], shade=True)

# Verificação de outliers com BOXPLOT
sea.boxplot(dataset_titanic_train["VRDeck"])

# Dataset de treino: Substituição
dataset_titanic_train.loc[dataset_titanic_train.VRDeck.isnull(),"VRDeck"] = dataset_titanic_train["VRDeck"].median()

# Dataset de teste: Substituição
dataset_titanic_test.loc[dataset_titanic_test.VRDeck.isnull(),"VRDeck"] = dataset_titanic_test["VRDeck"].median()

# Dataset de treino: Verificação
dataset_titanic_train.isnull().sum()

# Dataset de teste: Verificação
dataset_titanic_test.isnull().sum()

# # Dataset de treino: Substituição
# dataset_titanic_train["Transported"] = dataset_titanic_train["Transported"].astype(int)
# dataset_titanic_train

#**Datasets com colunas numéricas:**

# Datasets com colunas numéricas
dataset_titanic_train_numerica = dataset_titanic_train.select_dtypes(include=["float64","int64","bool"])
dataset_titanic_test_numerica = dataset_titanic_test.select_dtypes(include=["float64"])

# Datasets: verificação
dataset_titanic_train_numerica.info()
dataset_titanic_test_numerica.info()

#**Preparação dos Dados:**

from sklearn.model_selection import train_test_split

#**Definindo as variáveis de entrada e alvo:**

# Input variables
X = dataset_titanic_train_numerica.drop(["Transported"],axis=1)

# Target variable
y = dataset_titanic_train.Transported

# Treino e validação
X_train, X_val,  y_train, y_val = train_test_split(X, y, test_size = 0.20 , random_state = 42)

#**Modelos de Classificação:**

**Regressão Logística:**

# Importação
from sklearn.linear_model import LogisticRegression

# Criação do classificador
clf_rl = LogisticRegression(random_state=0).fit(X, y)

# Data Fit
clf_rl = clf_rl.fit(X_train,y_train)

# Previsão
y_pred_rl = clf_rl.predict(X_val)

#**Árvore de Decisão:**

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Criação do classificador
clf_tree = DecisionTreeClassifier(random_state=0).fit(X, y)

# Data Fit
clf_tree = clf_tree.fit(X_train,y_train)

# Previsão
y_pred_tree = clf_tree.predict(X_val)

#**Avaliação dos Modelos de Classificação:**

from sklearn.metrics import accuracy_score

**Accuracy:**

# Accuracy: Regressão Logística
accuracy_rl = accuracy_score(y_val,y_pred_rl)

# Accuracy: Árvore de Decisão
accuracy_tree = accuracy_score(y_val,y_pred_tree)

from sklearn.metrics import confusion_matrix

# Matriz de Confusão : Regressão Logística
confusion_matrix(y_val, y_pred_rl)

# Matriz de Confusão : Árvore de Decisão
confusion_matrix(y_val, y_pred_tree)

**Tabela de Performance:**

performance = pd.DataFrame({
    "Modelos": ["Regressão Logística", "Árvore de Decisão"],
    "Accuracy": [accuracy_rl, accuracy_tree]
})
performance

#**Previsão para a Base de Teste:**

X_teste = dataset_titanic_test_numerica

y_pred = clf_rl.predict(X_teste)

dataset_titanic_test_numerica["Transported"] = y_pred

dataset1_final_spaceship = pd.DataFrame({
    "PassengerId":dataset_titanic_test["PassengerId"],
    "Transported":dataset_titanic_test_numerica["Transported"]
})

dataset1_final_spaceship.to_csv("dataset_final.csv",index=False)

dataset1_final_spaceship
