# Exercise 4.18
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "Deep learning is a subset of machine learning. It uses neural networks with many layers."

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)
