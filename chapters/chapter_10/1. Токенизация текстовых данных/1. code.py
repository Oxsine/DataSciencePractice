from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

# Определение входного текста 
input_text = "Do you know how tokenization works? It's actually  quite interesting! Let's analyze а couple of sentences and  figure it out."

# Токенизация предложений  
print("\nSentence tokenizer:")  
print(sent_tokenize(input_text))

#Токенизатор слов  
print("\nWord tokenizer:")  
print(word_tokenize(input_text))

# Токенизатор пунктуации  
print("\nWord punct tokenizer:" )  
print(WordPunctTokenizer() .tokenize(input_text ))
