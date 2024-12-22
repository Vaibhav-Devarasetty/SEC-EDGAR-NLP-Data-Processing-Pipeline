from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def chunk_domain(text, k=5):
    candidate_labels = ['Design', 'Marketing', 'Education', 'Legal', 'Customer Service', 'Finance', 'Engineering', 'Human Resources', 'Sales', 'Health','Supply Chain/Logistics']

    domain_data = classifier(text, candidate_labels, multi_label=True)

    domain_labels = domain_data["labels"]
    domain_scores = domain_data["scores"]
    
    return domain_labels, domain_scores

text = "hello, bro i am a salesman"
labels, _ = chunk_domain(text)

from nltk.tokenize import sent_tokenize, word_tokenize
# Example usage:
file_name = "AIG_0000005272_10K_20231231_10-K_excerpt.txt"

with open(file_name, 'r') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Function to combine sentences until a chunk with at least 50 words is formed
def combine_sentences(sentences):
    combined_chunk = ''
    for sentence in sentences:
        if len(word_tokenize(sentence)) < 50:
            combined_chunk += sentence + ' '
        else:
            if combined_chunk:
                yield combined_chunk.strip()
                combined_chunk = ''
            yield sentence
    if combined_chunk:
        yield combined_chunk.strip()

# Combine sentences into chunks with at least 50 words
chunks = list(combine_sentences(sentences))

candidate_labels = ['Design', 'Marketing', 'Education', 'Legal', 'Customer Service', 'Finance', 'Engineering', 'Human Resources', 'Sales', 'Health','Supply Chain/Logistics']
best_chunks_per_domain = {}

for label in candidate_labels:
    best_chunks_per_domain[label] = []

threshold_value = 0.7

for chunk in chunks:
    domain_labels, domain_score = chunk_domain(chunk)
    if domain_score[0] >= threshold_value:
        best_chunks_per_domain[domain_labels[0]].append(chunk)

import pickle

# Specify the file path where you want to save the pickle file
file_path = "best_chunks_per_domain.pkl"

# Open the file in binary write mode and save the object using pickle.dump()
with open(file_path, "wb") as f:
    pickle.dump(best_chunks_per_domain, f)