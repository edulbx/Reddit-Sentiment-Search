#script - API/ML - Reedit - Others

import re #regularexpression
import praw #api
import config #api
import numpy as np #modeltrain

#packges from sklearn
from sklearn.model_selection import train_test_split #divison test
from sklearn.feature_extraction.text import TfidfVectorizer #vector with text
from sklearn.decomposition import TruncatedSVD #reduce dimension
from sklearn.neighbors import KNeighborsClassifier #ml model
from sklearn.ensemble import RandomForestClassifier #ml model
from sklearn.linear_model import LogisticRegressionCV #ml model
from sklearn.metrics import classification_report #model assessment
from sklearn.pipeline import Pipeline #...
from sklearn.metrics import confusion_matrix #print final result

import matplotlib.pyplot as plt #graphics
import seaborn as sns #graphics

#load
#subjects for search - variable target
assuntos = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']


#function begining
def load_data():

    #api reddit
    api_reddit = praw.Reddit (
        client_id = '-VEF-_JVUFfEDaaNEIlOEw',
        client_secret = 'htwLJm2R9vfx2Mlad8PkAVEvzTGPag',
        passwd = 'reedit123',
        user_agent = 'App-edu-reedit',
        username = 'Accomplished-Top6284'
    )

    #regex
    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext)) #função anonima incluida dentro da variavel
    
    #postfilter
    mask = lambda post: char_count(post) >= 100

    #result lists
    data = []
    labels = []

    #loop
    for i, assunto in enumerate(assuntos):

        # Extrai os posts
        subreddit_data = api_reddit.subreddit(assunto).new(limit = 1000)

        # Filtra os posts que não satisfazem nossa condição
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        # Adiciona posts e labels às listas
        data.extend(posts)
        labels.extend([i] * len(posts))

        # Print
        print(f"Número de posts do assunto r/{assunto}: {len(posts)}",
                f"\nUm dos posts extraídos: {posts[0][:600]}...\n",
              "_" * 80 + '\n')
    return data, labels
        
##this part is for teste train division:

#control variables
TEST_SIZE = .2
RANDOM_STATE = 0

#data split

def split_data():
    print(f"Split {100 * TEST_SIZE}% of data for training and model avaliation...")

    #spliting
    X_treino, X_teste, y_treino, y_teste = train_test_split(data, 
                                                            labels, 
                                                            test_size = TEST_SIZE, 
                                                            random_state = RANDOM_STATE)

    print(f"{len(y_teste)} amostras de testes.")

    return X_treino, X_teste, y_treino, y_teste



## pré-processamento de Dados e extração de atributos
# - Remove simbolos, numeros e strings semelahntes e a url com pre processador personalizado
# - vetoriza texto usando o termo frquencia inversa de frequencia de documento
# - reduz para valores principais usando decomposição de valor singular
# - particiona dados e rotulos em conjuntos de treinamento / validação

#variaveis de controle: 

MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

#função para pipeline de pré-processamento

def preprocessing_pipeline():

    #remove caracteres não alfabeticos
    pattern = r'\W|\d|http.*\s+|www.*\s+' 
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    #vetoriza TF-IDF
    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words = 'english', min_df = MIN_DOC_FREQ)
    
    #reduzindo a dimensionalidade da matriz tf-idf
    decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)

    #pipeline
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline


## vectorizer e o decomposition explicação Processamento de linguaguem natual
## o computador não entende texto só bits(passagem de corrente), tem que representar  texto de forma númerica, como uma tabela de frequencia:
## sentença/palavra | variável | resolver | de | problema | categórico
## 0                     1         0         1       1          0
## 1                     1         0         0       0          1  
## 2                     0         1         0       1          0
## essa mnatriz pode ser gerada com o count/vectorizer do skleran.
## pode-se ainda normalizar os dados dando importancias diferentes dentro de cada sentença ou documento, aplicando algum tipo de regra
## podemos fazer  isso com o TFIDFVectorizer do sklearn, exemplo com a sentença 1: 
## sentença 0; varivael = 0.51782; resolver = 0.0000; de 0.68239; problema = 051782; categórica = 0.0000.
## Isso pode ser feito TFIDFVectorizer. Na prática é isso que se entrega ao alogritmo, mas será retirada a palavra "de", através das stop words
## que são as palavras que aparecem com muita frequencia e não são importantes para o texto. Vai remover essas palavras com base nos dicionários
## de cada idioma, por isso o "stop_words= 'english'" deixando somente as palavras relvenates. E as demais palavras tabeladas com importancias
## diferentes. Cada uma com sua importancia, e pode acontecer de elas terem o mesmo peso a depender do caso.
## A palavra CountVectorizer implementa tokenização e contagem de ocorrência. 
## Tokenização é separar um paragrafo em frases e frases em palavras - etapa de pre processamento
## o TFIDFVectorizer combina todas as opções de CountVectorizer e TFIDFTransformer em um único model.
## Nós reponderamos vetores de recursos de contagem usando o método tf-idf e em seguida alimentamos os dados no classificador para uma melhor
## classificação. De outra forma as palavras comuns ocupariam muito espaço com pouca informação relevante sobre o texto.
## 
## USO DO TRUNCATE SVD na decomposition:
## A matriz pode ficar muito grande, e teria a maldição de dimensionalidade. Quanto maior o número de dimensões mais dificil
## para o algoritimo de ML. Cada operaçõa matematica que realizar vai consumir muito do computador então se aplica a redução da   
## dimensionalidade. Que é uma etapa de preprocessasmento. O Truncate SVD é um algoritimo de aprendizagem não supervisionada
##  Dai se cria um novo problema que precisa ser explicado em relaç]ao as matrizes.
## Matriz esparsa - tem muitos espaços com valores iguais a 0, como na Matriz acima; Outra opção seria a matriz densa.
## Na Matriz esparsa você acabaria tendo que fazer muitas operações com valor igual a zero, perdendo tempo e capacidade computacional.
## Não pode usar o PCA que é o mais usado normalmente, mas o correto nesse caso é o uso do TruncateSVD que é uma tecnica de redução da
## dimensionalidade (igual o PCA) e que usa o Singular Value Decomposition e partir dai aplica a redução da dimensionalide mesmo quando
## a matriz é esparsa. O PCA deixa a matriz mutio esparsa.


## função para criar models
#variaveis de controle: 
N_NEIGHBORS = 4 
CV = 3 #3 validações cruzadas

#função cria models: 
def cria_models():

    model_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    model_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    model_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    models = [("KNN", model_1), ("RandomForest", model_2), ("LogReg", model_3)]
    
    return models

# quando não se sabe previamente o melhor algoritmo tem que eexperiemntar algumas opções de algotirmos para sabe o ideal para determinado
# conjunto de dados.
# poderiam ser usados outros e para cada um dos algoritmos automatizar os hiperparametros.
# usando 4 vizinhos mais proximos, para experimentar. Pode ir tesntando e usar outros hiperparamtros
# usando 3 validações crizadas para o regresion CV.
# acima são apenas algorimos não o model em si

## TREINAMENTO E AVALIAÇÃO: 

#Função para trinamento e avaliação dos models
def treina_avalia(models, pipeline, X_treino, X_teste, y_treino, y_teste):
    
    results = []
    
    # Loop
    for name, model in models:

        # Pipeline
        pipe = Pipeline(pipeline + [(name, model)])

        # Treinamento
        print(f"Treinando o model {name} com dados de treino...")
        pipe.fit(X_treino, y_treino)

        # Previsões com dados de teste
        y_pred = pipe.predict(X_teste)

        # Calcula as métricas
        report = classification_report(y_teste, y_pred)
        print("Relatório de Classificação\n", report)
        with open('relatorio_classificacao.txt', 'w') as f:
                f.write(report)
        print('Salvo no arquivo')

        results.append([model, {'model': name, 'predictions': y_pred, 'report': report,}])           

    return results
#até aqui concluídas as funções que vamos precisar. 
#precisamos do bloco main, onde serão chamadas as funções e os results que foram criadas - herdado do C e C++.

#executando o pipeline de Machine Learning:

if __name__ == "__main__":
    #carregar os dados:
    data, labels = load_data() #recebe dois valores

    #faz a divisão
    X_treino, X_teste, y_treino, y_teste = split_data() #recebe 4 valores etc... tem que declarar as globais antes de chama a func

    #pipeline de pre-processamento: 
    pipeline = preprocessing_pipeline()

    #cria os models
    all_models = cria_models()

    #treina e avalia os results
    results = treina_avalia(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)
    # with open('relatorio_classificacao.txt', 'w') as f:
    #             f.write(results)

    print("Concluído com sucesso")

## Visualizando os results

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = "whitegrid")
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title("Number of Posts Per Subject")
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(),f"- {c} posts"]) for f,c in zip(assuntos, counts)])
    plt.show()

def plot_confusion(result):
    print("Classification Report\n", result[-1]['report'])
    y_pred = result[-1]['predictions']
    conf_matrix = confusion_matrix(y_teste, y_pred)
    _, test_counts = np.unique(y_teste, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(result[-1]['model'].upper() + " Results")
    plt.xlabel("Real Value")
    plt.ylabel("Model predictions")
    ticklabels = [f"r/{sub}" for sub in assuntos]
    plt.xticks(rotation=45, ha='right')
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f', cmap='viridis',
                    cbar=True
                                
                )
    plt.show()


# Gráfico de avaliação
plot_distribution()

# Resultado do KNN
plot_confusion(results[0])

# Resultado do RandomForest
plot_confusion(results[1])

# Resultado da Regressão Logística
plot_confusion(results[2])


# Fim
