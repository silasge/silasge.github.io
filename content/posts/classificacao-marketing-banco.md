---
author: "Silas Genário"
title: "Classificação de Marketing Bancário Usando o Sklearn"
date: "2022-08-21"
tags: ["Machine Learning", "Python", "Sklearn", "Estudos de Caso"]
draft: true
---

## Introdução

A Ciência de Dados tem um papel importante no processo de tomada de decisão de uma empresa. A alta disponibilidade de dados e a utilização de técnicas preditivas permitem às empresas otimizar seus custos com uma campanha ao focar nos clientes com maior potencial de rentabilidade.

Neste post farei uma aplicação desse tipo na área de marketing. A partir de uma campanha de um banco português que tinha como objetivo fazer com que clientes se subscreveram em uma conta de depósito a prazo[^1], aplico quatro modelos usando sklearn com o objetivo de prever quais clientes irão fazer o depósito.

[^1]: Depósito a Prazo é um tipo de produto bancário onde a instituição guarda os recursos do cliente por um prazo determinado (daí o nome) em troca de pagamento de juros. Aqui no Brasil seria semelhante à conta poupança ou CDB oferecido pelos bancos.

## Exploração dos Dados

Primeiro vamos carregar os pacotes necessários. 

``` py
import pandas as pd
from sklearn.model_selection import train_test_split
```

Podemos ler os dados com o Pandas e vamos dividi-los em conjuntos de treinamento e de teste usando a função `train_test_split` do sklearn.

``` py
bank = pd.read_csv("./data/raw/bank-additional-full.csv", sep=";")

bank_train, bank_test = train_test_split(bank, test_size=0.25, random_state=42)
```

Faremos a exploração dos dados no conjunto de treinamento, para evitar algum tipo de *leakage* durante a fase de modelagem.

Podemos explorar as primeiras observações dos dados e ver informações sobre o número de observações e tipos das colunas.


```python
bank_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>unknown</td>
      <td>unknown</td>
      <td>unknown</td>
      <td>telephone</td>
      <td>may</td>
      <td>tue</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.855</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>technician</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>thu</td>
      <td>...</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.964</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>basic.9y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.855</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>admin.</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>nov</td>
      <td>thu</td>
      <td>...</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.076</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>


```python
bank_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30891 entries, 0 to 30890
    Data columns (total 21 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             30891 non-null  int64  
     1   job             30891 non-null  object 
     2   marital         30891 non-null  object 
     3   education       30891 non-null  object 
     4   default         30891 non-null  object 
     5   housing         30891 non-null  object 
     6   loan            30891 non-null  object 
     7   contact         30891 non-null  object 
     8   month           30891 non-null  object 
     9   day_of_week     30891 non-null  object 
     10  duration        30891 non-null  int64  
     11  campaign        30891 non-null  int64  
     12  pdays           30891 non-null  int64  
     13  previous        30891 non-null  int64  
     14  poutcome        30891 non-null  object 
     15  emp.var.rate    30891 non-null  float64
     16  cons.price.idx  30891 non-null  float64
     17  cons.conf.idx   30891 non-null  float64
     18  euribor3m       30891 non-null  float64
     19  nr.employed     30891 non-null  float64
     20  y               30891 non-null  object 
    dtypes: float64(5), int64(5), object(11)
    memory usage: 4.9+ MB

Ao todo são 21 variáveis contendo atributos sobre o cliente, sobre a campanha de marketing do banco e sobre o contexto socioeconômico, além da variável que desejamos prever, `y`.

Não há observações nulas, de tal forma que não será necessário nenhum te imputação de dados faltantes, e os tipos das colunas parecem ser consistentes com seus valores.

É possível observar também que os dados são um pouco desbalanceados.

```python
bank_train["y"].value_counts(normalize=True)
```




    no     0.887119
    yes    0.112881
    Name: y, dtype: float64

Apenas cerca de 11% das observações são de clientes que decidiram pelo depósito a prazo. Na hora de avaliar os modelos, uma métrica como acurácia pode ser enganosa e não será representativa da qualidade de um modelo.

Podemos também oservar algumas estatísticas descritivas dos dados.

```python
bank_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
      <td>30891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.017287</td>
      <td>257.551746</td>
      <td>2.568839</td>
      <td>963.029361</td>
      <td>0.172348</td>
      <td>0.083264</td>
      <td>93.577223</td>
      <td>-40.506782</td>
      <td>3.622596</td>
      <td>5167.037687</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.448075</td>
      <td>259.371218</td>
      <td>2.747802</td>
      <td>185.544213</td>
      <td>0.492298</td>
      <td>1.570746</td>
      <td>0.579333</td>
      <td>4.629842</td>
      <td>1.734393</td>
      <td>72.461175</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.400000</td>
      <td>92.201000</td>
      <td>-50.800000</td>
      <td>0.634000</td>
      <td>4963.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>-1.800000</td>
      <td>93.075000</td>
      <td>-42.700000</td>
      <td>1.344000</td>
      <td>5099.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>38.000000</td>
      <td>179.000000</td>
      <td>2.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>93.749000</td>
      <td>-41.800000</td>
      <td>4.857000</td>
      <td>5191.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>47.000000</td>
      <td>318.000000</td>
      <td>3.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>93.994000</td>
      <td>-36.400000</td>
      <td>4.961000</td>
      <td>5228.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>98.000000</td>
      <td>4918.000000</td>
      <td>43.000000</td>
      <td>999.000000</td>
      <td>7.000000</td>
      <td>1.400000</td>
      <td>94.767000</td>
      <td>-26.900000</td>
      <td>5.045000</td>
      <td>5228.100000</td>
    </tr>
  </tbody>
</table>
</div>

Somente pela descrição das variáveis, é possível observar que algumas delas contém óbvios outliers ou são muito concentradas em alguns valores: a exemplo de `pdays` e `previous` onde as observações estão concentradas  em 999 e 0 respectivamente. A pouca informação contida nessas variáveis provavelmente não ajudará muito no que desejamos prever.

Também podemos comparar como as variáveis numéricas se comparam entre as classes da variável de resposta.

```python
bank_train.groupby("y").mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>39.932966</td>
      <td>220.495986</td>
      <td>2.637352</td>
      <td>984.727157</td>
      <td>0.132134</td>
      <td>0.252160</td>
      <td>93.605344</td>
      <td>-40.595333</td>
      <td>3.814645</td>
      <td>5176.265304</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>40.679954</td>
      <td>548.769429</td>
      <td>2.030399</td>
      <td>792.508460</td>
      <td>0.488385</td>
      <td>-1.244078</td>
      <td>93.356221</td>
      <td>-39.810869</td>
      <td>2.113300</td>
      <td>5094.518727</td>
    </tr>
  </tbody>
</table>
</div>

Destaques para `duration`