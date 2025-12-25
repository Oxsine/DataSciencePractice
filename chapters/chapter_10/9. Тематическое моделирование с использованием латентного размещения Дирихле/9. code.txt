from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

# Загрузка входных данных
def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])
    return data

# Функция обработки, предназначенная для токенизации текста,
# удаления стоп-слов и выполнения стемминга
def process(input_text):
    # Создание регулярного выражения для токенизатора
    tokenizer = RegexpTokenizer(r'\w+')

    # Создание стеммера Сноуболла
    stemmer = SnowballStemmer('english')

    # Получение списка стоп-слов
    stop_words = stopwords.words('english')

    # Токенизация входной строки
    tokens = tokenizer.tokenize(input_text.lower())

    # Удаление стоп-слов
    tokens = [x for x in tokens if not x in stop_words]

    # Выполнение стемминга токенизированных слов
    tokens_stemmed = [stemmer.stem(x) for x in tokens]
    return tokens_stemmed

if __name__ == '__main__':
    # Загрузка входных данных

    sample_text = """Machine learning is a subset of artificial intelligence.
    Natural language processing helps computers understand human language.
    Python is a popular programming language for data science.
    Deep learning uses neural networks for complex tasks."""
    
    with open('data.txt', 'w') as f:
        f.write(sample_text)
        
    data = load_data('data.txt')

# Создание списка токенов предложений
tokens = [process(x) for x in data]

# Создание словаря на основе токенизированных предложений
dict_tokens = corpora.Dictionary(tokens)

# Создание терм-документной матрицы
doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]

# Определим количество тем для LDA-модели
num_topics = 2

# Генерирование LDA-модели
ldamodel = models.ldamodel.LdaModel(doc_term_mat,
          num_topics=num_topics, id2word=dict_tokens, passes=25)

num_words = 5
print('\nTop' + str(num_words) + ' contributing words to each topic:')
for item in ldamodel.print_topics(num_topics=num_topics,                   
                                  num_words=num_words):
    print('\nTopic', item[0])

    # Вывод представительных слов вместе с их
    # относительными вкладами
    list_of_strings = item[1].split(' + ')
    for text in list_of_strings:
        weight = text.split('*')[0]
        word = text.split('*')[1]
        print(word, '==>', str(round(float(weight) * 100, 2)) + '%')
