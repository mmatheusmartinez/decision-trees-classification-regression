
# Árvores de Decisão - Classificação e Regressão

Dois projetos práticos com árvores de decisão usando scikit-learn: um de classificação (prever presença de doença cardíaca) e outro de regressão (prever o tempo de entrega de pedidos da Amazon). Os dois seguem o mesmo fluxo - entender os dados, preparar as variáveis, treinar uma árvore sem restrições, diagnosticar overfitting, podar, validar com cross-validation e ajustar hiperparâmetros com Grid Search.

A ideia é mostrar não só o modelo final, mas o caminho até ele: por que uma árvore sem limite de profundidade decora o treino, como a poda muda o comportamento e como a busca de hiperparâmetros ajuda a encontrar um ponto de equilíbrio entre viés e variância.

---

## Estrutura do repositório

```
.
├── classificacao_doenca_cardiaca.py
├── regressao_tempo_entrega.py
├── requirements.txt
└── README.md
```

---

## Projeto 1 - Classificação: predição de doença cardíaca

### Problema

Dado um conjunto de exames e características clínicas de pacientes, prever se há presença (`Presence`) ou ausência (`Absence`) de doença cardíaca.

### Dados

[Heart Disease Prediction](https://raw.githubusercontent.com/vqrca/ml-datasets/refs/heads/main/Heart_Disease_Prediction.csv) - 270 pacientes e 13 variáveis clínicas, como idade, sexo, tipo de dor no peito, pressão arterial, colesterol, frequência cardíaca máxima e resultado do exame de tálio.

### O que foi feito

Os dados passaram por one-hot encoding nas variáveis `Chest pain type` e `Thallium`, que são categóricas apesar de estarem codificadas como números. A divisão treino/teste (80/20) foi estratificada para manter a proporção entre as classes.

A primeira árvore foi treinada sem nenhuma restrição, o que serviu de referência para evidenciar o overfitting - acurácia perfeita no treino e queda considerável no teste. A partir daí o modelo foi podado com `max_depth=3` e avaliado com:

- Acurácia, precisão, recall e F1-score (`classification_report`)
- Matriz de confusão normalizada
- Curva ROC e AUC
- Validação cruzada simples e estratificada (`StratifiedKFold`), com média e desvio padrão das métricas

Por fim, um `GridSearchCV` testou combinações de `max_depth`, `min_samples_leaf` e `criterion` (gini e entropy), e as três versões do modelo (inicial, podada e otimizada) foram comparadas lado a lado pelas curvas ROC.

---

## Projeto 2 - Regressão: previsão do tempo de entrega

### Problema

Estimar o tempo de entrega (em minutos) de pedidos a partir de informações do entregador, da rota, do clima, do trânsito e do tipo de produto.

### Dados

[Amazon Delivery Dataset](https://raw.githubusercontent.com/vqrca/ml-datasets/refs/heads/main/amazon_delivery.csv) - pedidos realizados na Índia, com coordenadas da loja e do cliente, horários de pedido e retirada, avaliação do entregador, clima, trânsito, veículo, tipo de área e categoria do produto.

### Preparação e engenharia de atributos

Essa foi a parte mais trabalhosa do projeto. A limpeza removeu registros com coordenadas fora do território indiano, avaliações de entregador inválidas e valores ausentes em clima. Duas variáveis novas foram criadas:

- **`Distancia_km`** - distância entre loja e cliente calculada pela fórmula de Haversine, substituindo as quatro colunas de latitude e longitude.
- **`Tempo_Preparo_min`** - intervalo entre o horário do pedido e o da retirada, com correção para pedidos que atravessam a meia-noite.

As variáveis categóricas (`Weather`, `Traffic`, `Vehicle`, `Area`, `Category`) passaram por one-hot encoding, e identificadores, coordenadas e horários brutos foram descartados antes do treino.

### Modelagem e avaliação

Assim como no projeto de classificação, a primeira árvore foi treinada sem restrições para expor o overfitting, comparando RMSE e R² entre treino e teste. Depois veio a poda com `max_depth=10` e, na sequência, um `GridSearchCV` com validação cruzada de 5 partições otimizando o RMSE sobre `max_depth`, `min_samples_leaf` e `criterion` (squared_error e absolute_error).

Métricas usadas: MAE, MSE, RMSE e R².

---

## Tecnologias

- Python 3
- pandas e NumPy
- scikit-learn
- Matplotlib

---

Os datasets são carregados direto do GitHub, então não é preciso baixar nada. Os scripts estão divididos em células (`#%%`), o que permite rodar bloco a bloco no Spyder ou no VS Code com a extensão Python.

---

## Principais aprendizados

Árvores de decisão sem restrições têm uma tendência forte a overfitting, e isso ficou claro nos dois problemas - o modelo praticamente decora o treino e generaliza mal. Limitar profundidade e exigir um mínimo de amostras por folha reduz bastante essa distância entre treino e teste, às vezes com ganho real no teste. Também ficou evidente o quanto a engenharia de atributos pesa no resultado: no projeto de regressão, transformar coordenadas em distância e horários em tempo de preparo deu ao modelo variáveis com sentido prático, em vez de números soltos.

---


