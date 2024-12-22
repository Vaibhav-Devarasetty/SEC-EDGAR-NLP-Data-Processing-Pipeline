import os
import sys
from datetime import datetime, timedelta
# Use a pipeline as a high-level helper
from transformers import BartTokenizerFast, BartForConditionalGeneration, pipeline
import torch
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

device = torch.device("cpu")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model.to(device)
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def domain_predict(text):
    candidate_labels = ['Design', 'Marketing', 'Education', 'Legal', 'Customer Service', 'Finance', 'Engineering', 'Human Resources', 'Sales', 'Health','Supply Chain/Logistics']

    domain = classifier(text, candidate_labels)["labels"][0]

    return domain

def pipe(text):
    inputs = tokenizer([text], return_tensors="pt").to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], attention_mask=attention_mask, num_beams=2, min_length=0, max_length=5000)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return summary

def save_chunks_to_files(chunks, original_file_path):
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    summary_file_path = os.path.join(os.path.dirname(original_file_path), f"{base_name}_summary.txt")
    
    with open(summary_file_path, 'w') as file:
        for chunk in chunks:
            file.write(chunk + "\n\n")
    
    print(f"Chunks have been saved to {summary_file_path}")
    return chunks

def load_summaries_from_file(file_path):
    with open(file_path, 'r') as file:
        summaries = file.read().split("\n\n")
    return summaries

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_company(ticker, all_companies_file):
    with open(all_companies_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if len(line.split()) == 2:
                cik, company_ticker = line.split()
                if company_ticker == ticker:
                    with open(f'{os.getcwd()}/SEC-EDGAR-text/company_info.txt', 'w') as output_file:
                        output_file.write(line)
                    print(f"Company information for '{ticker}' has been extracted to '{ticker}_company_info.txt'.")
                    return
        print(f"No company found with ticker '{ticker}'.")

def find_text_files(directory, ticker, filing_type):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.endswith(".txt") and
                ticker.upper() in file.upper() and
                filing_type.upper() in file.upper() and
                not file.upper().endswith("_SUMMARY.TXT")
            ):
                matching_files.append(os.path.join(root, file))
    return matching_files


def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def summarize_chunks(chunks, file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(os.path.dirname(file_path), f"{base_name}_summary.txt")

    summaries = []
    if os.path.exists(summary_file_path):
        print(f"Summary file {summary_file_path} already exists. Loading summaries from the file.")
        summaries = load_summaries_from_file(summary_file_path)
        
    else:
        summaries = []
        for chunk in chunks:
            summary = pipe(chunk)
            #print(summary)
            summaries.append(summary)

        save_chunks_to_files(summaries, file_path)

    return summaries, summary_file_path

def make_file_from_chunks(texts, file_path, domain):
    with open(file_path, 'w') as file:
            file.write("\n")

    # Save each text to a separate .txt file
    for text in texts:
        #if domain_predict(text) == domain:
            with open(file_path, 'a') as file:
                file.write(text)

    print("Files have been saved successfully.")

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    client = OpenAI(api_key='sk-proj-rnvImRGIkwtlOigie9LHT3BlbkFJyvbASU5Iqj9yyPewC5bw')
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def create_vector_store(file_paths):
    client = OpenAI(api_key='sk-proj-rnvImRGIkwtlOigie9LHT3BlbkFJyvbASU5Iqj9yyPewC5bw')
    vector_store = client.beta.vector_stores.create(name="Company Filings")

    fps = []
    for path in file_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]
        summary_file_path = os.path.join(os.path.dirname(path), f"{base_name}_summary.txt")
        fps.append(summary_file_path)
    
    file_streams = [open(path, "rb") for path in fps]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )

    print(file_batch.status)
    print(file_batch.file_counts)
    return vector_store

def perform_semantic_search(vector_store, key_phrases):
    results = []
    for key_phrase in key_phrases:
        embedding = get_embedding(key_phrase)
        response = vector_store.search(embedding, max_results=5)
        results.append(response)
    return results

def quote_original_text(results, original_chunks, key_phrases):
    quotes = []
    for i, result in enumerate(results):
        for match in result['data']:
            chunk_index = match['metadata']['chunk_index']
            original_text = original_chunks[chunk_index]
            quotes.append({
                "key_phrase": key_phrases[i],
                "quote": original_text,
                "relevance": match['score']
            })
    return quotes

def combine_text_files(file_paths, output_file_path):
    """
    Combines the content of multiple text files into a single text file.
    
    Parameters:
    - file_paths: List of paths to the input text files.
    - output_file_path: Path to the output text file.
    """
    try:
        with open(output_file_path, 'w') as output_file:
            for file_path in file_paths:
                with open(file_path, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
                    output_file.write("\n")  # Optionally add a newline between files
        print("Files have been combined successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def pipeline(company_name, ticker, key_phrases, domain, directory_path="./SEC-EDGAR-text"):
    ensure_directory_exists(directory_path)
    matching_files_10_K = find_text_files(directory_path, ticker, "10-K")
    matching_files_10_Q = find_text_files(directory_path, ticker, "10-Q")

    if not (matching_files_10_K and matching_files_10_Q):
        extract_company(ticker=ticker, all_companies_file=f"{os.getcwd()}/SEC-EDGAR-text/all_companies_list.txt")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        command = f"python SEC-EDGAR-text --companies_list={os.getcwd()}/SEC-EDGAR-text/company_info.txt --storage={os.getcwd()}/SEC-EDGAR-text/sec_edgar_new --start={start_date} --end={end_date} --filings=10-K,10-Q --report_period=all"
        os.system(command)
        matching_files_10_K = find_text_files(directory_path, ticker, "10-K")
        matching_files_10_Q = find_text_files(directory_path, ticker, "10-Q")

    if not (matching_files_10_K and matching_files_10_Q):
        print(f"The Ticker {ticker} is not listed in the SEC EDGAR server.")
        return

    required_files = [matching_files_10_K[-1], matching_files_10_Q[-1], matching_files_10_Q[-2], matching_files_10_Q[-3]]
    
    for file_path in required_files:
        file_name = file_path

        with open(file_name, 'r') as file:
            text = file.read()

        start_index = text.find("UNITED STATES")

        if start_index != -1:
            required_text = text[start_index:]

            with open(file_name, 'w') as output_file:
                output_file.write(required_text)
                #print("Extracted text has been written to extracted_text.txt")
        else:
            print("Starting text not found in the file.")

    summarized_chunks = []
    original_chunks = []
    summarised_file_paths = []

    for file_path in required_files:
        with open(file_path, 'r') as file:
            text = file.read()
        chunks = list(chunk_text(text))
        original_chunks.extend(chunks)
        chunks_of_summary, summary_file_path = summarize_chunks(chunks, file_path)
        summarized_chunks.extend(chunks_of_summary)
        summarised_file_paths.append(summary_file_path)
    
    domain_summary_file = f"{ticker}_{company_name}_{domain}_langchain_summary_search.txt"
    
    related_chunks = []
    for original_chunk in original_chunks:
        if domain == domain_predict(original_chunk):
            related_chunks.append(original_chunk)
    
    with open(domain_summary_file, 'w') as file:
        for chunk in related_chunks:
            file.write(chunk + "\n\n")

    os.environ['OPENAI_API_KEY'] = "sk-proj-rnvImRGIkwtlOigie9LHT3BlbkFJyvbASU5Iqj9yyPewC5bw"

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader(domain_summary_file).load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = text_splitter.split_documents(raw_documents)
    # db = Chroma.from_documents(documents, OpenAIEmbeddings())
    
    # results_dir = f"./results/{ticker}"
    # ensure_directory_exists(results_dir)
    # for key_phrase in key_phrases:
    #     print("---------------------------------------------------------------------------------")
    #     docs = db.similarity_search(key_phrase)
    #     print(f"Key Phrase: {key_phrase}\n\n")
    #     print(f"\t{docs[0].page_content}")

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = db.as_retriever()

    
    results_dir = f"./results/{ticker}"
    ensure_directory_exists(results_dir)
    for key_phrase in key_phrases:
        print("---------------------------------------------------------------------------------")
        docs = retriever.invoke(key_phrase)
        print(f"Key Phrase: {key_phrase}\n\n")
        print(f"\t{docs[0].page_content}")




key_phrases =  ["Mindtickle's sales enablement platform offers comprehensive training programs designed to improve sales performance. It provides tools for onboarding, continuous learning, and skill development, enabling sales teams to stay updated and effective. The platform uses personalized learning paths, interactive content, and analytics to track progress and ensure sales readiness.", 
                "Mindtickle's Sales Content Management Solutions provide a centralized platform for managing and distributing sales content. The service enables organizations to store, organize, and share sales materials such as presentations, case studies, and brochures. It ensures that sales teams have easy access to the most up-to-date content, supports content personalization, and tracks content usage and effectiveness. This helps improve sales productivity, consistency in messaging, and overall sales performance.",
                "Mindtickle's Sales Coaching Software is designed to enhance sales team performance through structured coaching programs. It offers features such as skill assessments, personalized learning paths, feedback mechanisms, and analytics to monitor progress. This software helps managers deliver targeted coaching, track improvements, and ensure that sales representatives are well-prepared and continuously developing their skills.",
                "Mindtickle's Sales Readiness Index analyzes sales team performance to ensure sales readiness. The platform provides insights into the effectiveness of sales training, coaching, and content usage. It helps organizations measure skill levels, identify knowledge gaps, and track improvements over time. This data-driven approach enables sales leaders to make informed decisions and optimize their team's performance for better results.",
                "Mindtickle's Conversation Intelligence platform analyzes sales calls to provide insights for improving sales performance. It uses AI to transcribe and evaluate conversations, identifying key trends, customer sentiments, and areas for improvement. This helps sales teams refine their messaging, understand customer needs better, and enhance overall sales effectiveness.",
                "Mindtickle's Revenue Intelligence platform provides insights into sales performance and revenue trends. It integrates data from various sources to offer a comprehensive view of sales activities, pipeline health, and forecast accuracy. This helps sales leaders make informed decisions, identify opportunities, and address challenges to drive revenue growth.",
                "Mindtickle's Digital Sales Rooms offer a personalized virtual space for sales teams to engage with prospects and customers. These rooms provide a centralized location for sharing content, tracking engagement, and facilitating communication throughout the sales process. This tool aims to enhance the buyer's experience, streamline interactions, and improve the effectiveness of sales engagements.",
                "Mindtickle's Sales Enablement Analytics Software provides insights into the effectiveness of sales training and content. It tracks engagement, completion rates, and performance metrics to help sales leaders understand the impact of their enablement programs. The platform uses data-driven analysis to identify areas for improvement, optimize training strategies, and ensure that sales teams are well-prepared to meet their targets."]


if __name__ == '__main__':
    tickers = ["CSCO", "AIG", "AMD", "NVDA"]
    company_name = "MindTickle_summary"

    key_phrases_string = ""

    for key_phrase in key_phrases:
        key_phrases_string += key_phrase

    domain = domain_predict(key_phrases_string)

    for ticker in tickers:
        results_dir = f"./results/{ticker}"
        ensure_directory_exists(results_dir)
        output_file_path = f'{results_dir}/{company_name}_{ticker}_domain({domain})_summary_based_lang_chain_search.txt'
        with open(output_file_path, "w") as f:
            # Store the reference to the original standard output
            original_stdout = sys.stdout

            # Redirect the standard output to the file
            sys.stdout = f
            
            # Your main script code here
            pipeline(company_name=company_name, ticker=ticker, key_phrases=key_phrases, domain=domain)

            # Restore the original standard output
            sys.stdout = original_stdout
