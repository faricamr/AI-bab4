# Exercise 4.19
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    "This is a good movie.",                              # positif
    "I love this product, it works perfectly!",           # positif
    "This restaurant serves amazing food.",               # positif
    "The service was terrible and the food was cold.",    # negatif
    "I hate waiting in long lines.",                      # negatif
    "This is the worst experience I've ever had.",        # negatif
    "The movie was okay, nothing special.",               # netral / ambigu
    "It's not bad, but not great either.",                # netral / ambigu
]

for s in sentences:
    result = classifier(s)
    print("{:60s} -> {}".format(s, result[0]))