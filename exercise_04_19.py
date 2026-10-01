# Exercise 4.19
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('This is a terrible movie.'))
print(classifier('I love this product!'))
print(classifier('I hate waiting in line.'))
