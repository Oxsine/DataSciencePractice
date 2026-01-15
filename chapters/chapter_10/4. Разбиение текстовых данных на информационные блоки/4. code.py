import numpy as np  
from nltk.corpus import brown 

# Разбиение входного текста на блоки,  
# причем каждый блок содержит N слов  
def chunkier(input_data, N):  
    input_words = input_data.split(' ')  
    output = []

    curchunk = [ ]  
    count = 0  
    for word in input_words:  
        curchunk.append(word)  
        count += 1  
        if count == N:  
            output.append(' '.join(curchunk))  
            count, curchunk = 0,[] 
    output.append(' '.join(curchunk)) 
    return output 

if True:
    # Чтение первых 12000 слов из коллекции Brown  
    input_data = ' '.join(brown.words() [:12000]) 

    # Определение количества слов в каждом блоке  
    chunksize = 700  

    chunks = chunkier(input_data, chunksize)  
    print('\nNumЬer of text chunks =', len(chunks), '\n')  
    for i, chunk in enumerate(chunks):  
        print('Chunk', i+1, '==>', chunk[:50]) 
