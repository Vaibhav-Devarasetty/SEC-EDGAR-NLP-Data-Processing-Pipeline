from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from transformers import BartTokenizerFast, BartForConditionalGeneration, pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download the necessary resources for NLTK
nltk.download('punkt')

device = torch.device("cpu")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model.to(device)
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def chunk_domain(text, k=5):
    candidate_labels = ['Design', 'Marketing', 'Education', 'Legal', 'Customer Service', 'Finance', 'Engineering', 'Human Resources', 'Sales', 'Health','Supply Chain/Logistics']

    domain_data = classifier(text, candidate_labels, multi_label=True)

    domain_labels = domain_data["labels"]
    domain_scores = domain_data["scores"]
    for i in range(len(candidate_labels)):
        print(f"{domain_labels[i]}: {domain_scores[i]}")

def pipe(text):
    inputs = tokenizer([text], return_tensors="pt").to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], attention_mask=attention_mask, num_beams=2, min_length=0, max_length=5000)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return summary

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Example usage:
file_name = "AIG_0000005272_10K_20231231_10-K_excerpt.txt"

with open(file_name, 'r') as file:
    text = file.read()

#chunks = text_splitter.split_text(text)
# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Function to combine sentences until a chunk with at least 20 words is formed
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

# Combine sentences into chunks with at least 20 words
chunks = list(combine_sentences(sentences))

output_file_path = f'AIG_chunks_summarised_domain.txt'
with open(output_file_path, "w") as f:
    # Store the reference to the original standard output
    original_stdout = sys.stdout

    # Redirect the standard output to the file
    sys.stdout = f
    
    # Your main script code here
    for chunk in chunks:
        print("--------------------------------------------------------------------------------")
        print("ORIGINAL CHUNK")
        print(chunk)
        print()
        print("SUMMARISED CHUNK")
        print(pipe(chunk))
        print()
        print("CLASSIFICATION OF THE CHUNK")
        print(chunk_domain(chunk))
        

    # Restore the original standard output
    sys.stdout = original_stdout