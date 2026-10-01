# Exercise 4.20
from transformers import pipeline

question_answerer = pipeline('question-answering')

context1 = """
Python is a high-level, general-purpose programming language created by
Guido van Rossum and first released in 1991. It emphasizes code readability
with significant indentation. Python is dynamically typed and garbage-collected.
"""

questions1 = [
    'Who created Python?',
    'When was Python first released?',
    'What does Python emphasize?',
    'What type of programming language is Python?',
]

print("=== Context 1: Python ===")
for q in questions1:
    result = question_answerer({'question': q, 'context': context1})
    print("{:40s} -> {}".format(q, result['answer']))

context2 = """
Indonesia is a country in Southeast Asia and Oceania, between the Indian
and Pacific oceans. It consists of over 17,000 islands, including Sumatra,
Java, Sulawesi, and parts of Borneo and New Guinea. Indonesia is the world's
largest archipelagic state and the 14th-largest country by land area.
The capital is Jakarta, and the official language is Indonesian.
"""

questions2 = [
    'Where is Indonesia located?',
    'How many islands does Indonesia have?',
    'What is the capital of Indonesia?',
    'What is the official language of Indonesia?',
]

print("\n=== Context 2: Indonesia ===")
for q in questions2:
    result = question_answerer({'question': q, 'context': context2})
    print("{:40s} -> {}".format(q, result['answer']))