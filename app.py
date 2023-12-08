from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dados = pd.read_csv('example.csv')

# dados = [
#     ["João", 8.5, 12, 3, 85, 1],    # 1 indica aprovado
#     ["Maria", 6.5, 8, 1, 75, 0],    # 0 indica reprovado
#     ["Carlos", 7.8, 15, 2, 90, 1],  # 1 indica aprovado
#     # Adicione mais alunos conforme necessário
# ]

# Converter dados para formato adequado
nomes = [linha[0] for linha in dados]
X = [[linha[1], linha[2], linha[3], linha[4]] for linha in dados]
y = [linha[5] for linha in dados]

# Dividir o conjunto de dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# Criar e treinar o modelo de Regressão Logística
modelo = LogisticRegression(random_state=42)
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_teste, previsoes)
print(f"Precisão do modelo: {precisao * 100:.2f}%")

# Exemplo de como usar o modelo para fazer previsões para novos alunos
novo_aluno = [[7.0, 10, 2, 80]]
novo_aluno_padronizado = scaler.transform(novo_aluno)
previsao_novo_aluno = modelo.predict(novo_aluno_padronizado)

if previsao_novo_aluno[0] == 1:
    print("O novo aluno será aprovado!")
else:
    print("O novo aluno não será aprovado.")