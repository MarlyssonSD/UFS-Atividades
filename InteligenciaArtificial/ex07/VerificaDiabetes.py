import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. Carregar os dados
data = pd.read_csv('diabetes.csv')

# 2. Definir características e rótulos
X = data.drop('Outcome', axis=1)  # Dados para treinamento
y = data['Outcome']                # Resultados

# 3. 100% dos dados são usados para o treinamento
model = Perceptron()
model.fit(X, y)

# 4. Carregar os dados para teste
test_data = pd.read_csv('test_diabetes.csv')

# 5. Separar as características e rótulos do arquivo de teste
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']  # Armazena os resultados esperados

# 6. Fazer previsões
predictions = model.predict(X_test)

# 7. Mostrar resultados detalhados
for i, prediction in enumerate(predictions):
    expected = y_test.iloc[i]
    print(f'Teste {i + 1}: Resultado Previsto = {prediction}, Resultado Esperado = {expected}')

# 8. Calcular a acurácia
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do teste: {accuracy * 100:.2f}%')