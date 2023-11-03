import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

assuntos = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']

TEST_SIZE = .2
RANDOM_STATE = 0
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30
N_NEIGHBORS = 4
CV = 3

def carrega_dados():
    api_reddit = praw.Reddit(
        client_id = "5QjXa_rfMIm-kyvMlZyfcg",
        client_secret = "0JqCti4Skz3_OGhI18e6HHNF4cbYcQ",
        password = "e7epzdvd",
        user_agent = "testscript by u/SMR_Sassa",
        username = "SMR_Sassa",
    )
    # Contagem do numero de caracteres
    char_count = lambda post: len(re.sub(r'\W|\d', '', post.selftext))
    # Só posts com mais de 100 caracteres
    mask = lambda post: char_count(post) >= 100

    data = []
    labels = []
    for i, assunto in enumerate(assuntos):
        subreddit_data = api_reddit.subreddit(assunto).new(limit=1000)
        posts = [post.selftext for post in filter(mask, subreddit_data)]
        data.extend(posts)
        labels.extend([i] * len(posts))
        print(f"Numero de posts do assunto r/{assunto}: {len(posts)}",
              f"\nUm dos posts extraidos: {posts[0][:600]}...\n",
              "_"*80+"\n")
    
    return data, labels

def split_data():
    print(f"Split {100 * TEST_SIZE} % dos dados para treinamento e avaliação dos modelo")
    # Split dos dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        data,
        labels,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE
    )

    print(f"{len(y_teste)} amostra de teste")

    return X_treino, X_teste, y_treino, y_teste

# Remove símbolos, números e strings semelhantes a url com pré-processador personalizado
# Vetoriza texto usando o termo frequência inversa de frequência de documento
# Reduz para valores principais usando decomposição de valor singular
# Particiona dados e rótulos em conjuntos de treinamento / validação
# Função para o pipeline de pré-processamento
def preprocessing_pipeline():
    # Remove caracteres não "alfabéticos"
    pattern = r'\W|\dZhttp.*\s+Zwww.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words = 'english', min_df = MIN_DOC_FREQ)

    # Reduzindo a dimensionalidade da matriz TF-IDF
    decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)

    # Pipeline
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline

def cria_modelos():
    modelo1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    modelo2 = RandomForestClassifier(random_state = RANDOM_STATE)
    modelo3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    modelos = [("KNN", modelo1), ("RandomForest", modelo2), ("LogReg", modelo3)]

    return modelos

def treina_modelos(modelos, pipeline, X_treino, X_teste, y_treino, y_teste):
    resultados = []
    for name, modelo in modelos:
        pipe = Pipeline(pipeline + [(name, modelo)])

        print(f"Treinando o modelo {name} com dados de treino...")
        # Treinamento
        pipe.fit(X_treino, y_treino)
        # Previsões com dados de teste
        y_pred = pipe.predict(X_teste)
        # Calcula metricas
        report = classification_report(y_teste, y_pred)
        print("Relatorio de Classificação\n", report)

        resultados.append([modelo, {"modelo": name, "previsoes": y_pred, "report": report}])

    return resultados

if __name__ == '__main__':
    data, labels = carrega_dados()
    X_treino, X_teste, y_treino, y_teste = split_data()
    pipeline = preprocessing_pipeline()
    all_models = cria_modelos()
    resultados = treina_modelos(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = "whitegrid")
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title("Número de Posts Por Assunto")
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(),f"- {c} posts"]) for f,c in zip(assuntos, counts)])
    plt.show()

def plot_confusion(result):
    print("Relatório de Classificação\n", result[-1]['report'])
    y_pred = result[-1]['previsoes']
    conf_matrix = confusion_matrix(y_teste, y_pred)
    _, test_counts = np.unique(y_teste, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(result[-1]['modelo'].upper() + " Resultados")
    plt.xlabel("Valor Real")
    plt.ylabel("Previsão do Modelo")
    ticklabels = [f"r/{sub}" for sub in assuntos]
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
    plt.show()

# Gráfico de avaliação
plot_distribution()
# Resultado do KNN
plot_confusion(resultados[0])
# Resultado do RandomForest
plot_confusion(resultados[1])
# Resultado da Regressão Logística
plot_confusion(resultados[2])
