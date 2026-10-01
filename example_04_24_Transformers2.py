# Example 4.24
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from transformers import pipeline

question_answerer = pipeline('question-answering')
print(question_answerer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
}))