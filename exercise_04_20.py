# Exercise 4.20
from transformers import pipeline

question_answerer = pipeline('question-answering')
print(question_answerer({
    'question': 'When was the company created?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
}))
