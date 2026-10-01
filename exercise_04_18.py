# Exercise 4.18

import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = """
Python is a high-level, general-purpose programming language.
Its design philosophy emphasizes code readability with the use of
significant indentation. Python is dynamically typed and garbage-collected.
It supports multiple programming paradigms, including structured,
object-oriented and functional programming. It was created by Guido van Rossum
and first released in 1991. Python is commonly used in artificial intelligence,
data science, and web development.
"""

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)