import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.corpus import brown 

# Определение класса TextChunker (если его нет в отдельном файле)
class TextChunker:
    def __init__(self, text, chunk_words):
        self.text = text
        self.chunk_words = chunk_words
        self.words = text.split()
        self.num_chunks = len(self.words) // chunk_words
        
    def __iter__(self):
        words = self.text.split()
        for i in range(0, len(words), self.chunk_words):
            yield ' '.join(words[i:i + self.chunk_words])
            
    def __len__(self):
        return self.num_chunks

# Чтение данных из коллекции Brown 
input_data = ' '.join(brown.words()[:5400]) 

# Количество слов в каждом блоке 
chunk_size = 800 

# Создание чанков
text_chunker = TextChunker(input_data, chunk_size)  # Используем класс

# Преобразование в элементы словаря 
chunks = [] 
for count, chunk in enumerate(text_chunker):  # Итерируем по объекту
    d = {'index': count, 'text': chunk} 
    chunks.append(d)

# Извлечение терм-документной матрицы 
count_vectorizer = CountVectorizer(min_df=7, max_df=20) 
document_term_matrix = count_vectorizer.fit_transform([chunk['text'] for chunk in chunks])

# Извлечение и отображение словаря 
vocabulary = np.array(count_vectorizer.get_feature_names_out())  # Изменено с get_feature_names()
print("\nVocabulary:\n", vocabulary)

# Генерация имен блоков 
chunk_names = [] 
for i in range(len(chunks)):  # Используем длину chunks, а не text_chunks
    chunk_names.append('Chunk-' + str(i+1))

# Вывод терм-документной матрицы 
print("\nDocument term matrix:") 
formatted_text = '{:>12}' * (len(chunk_names) + 1) 
print('\n', formatted_text.format('Word', *chunk_names), '\n') 

# Для корректного вывода нужно преобразовать матрицу в плотный формат
doc_matrix_dense = document_term_matrix.toarray().T  # Транспонируем для вывода по словам

for word, row in zip(vocabulary, doc_matrix_dense):
    output = [word] + [str(freq) for freq in row]
    print(formatted_text.format(*output))
