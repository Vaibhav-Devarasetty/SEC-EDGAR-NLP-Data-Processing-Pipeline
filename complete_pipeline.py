import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Download the necessary resources for NLTK
nltk.download('punkt')

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_company(ticker, all_companies_file):
    # Open the file containing the list of companies and their ticker values
    with open(all_companies_file, 'r') as file:
        # Read each line from the file
        lines = file.readlines()
        # Loop through each line to find the line with the matching ticker
        for line in lines:
            # Split the line into CIK and ticker
            if len(line.split()) == 2:
                cik, company_ticker = line.split()
                # Check if the ticker matches the input ticker
                if company_ticker == ticker:
                    # If a match is found, create a new file and write the line to it
                    with open(f'{os.getcwd()}/SEC-EDGAR-text/company_info.txt', 'w') as output_file:
                        output_file.write(line)
                    print(f"Company information for '{ticker}' has been extracted to '{ticker}_company_info.txt'.")
                    return
        # If no match is found, print a message
        print(f"No company found with ticker '{ticker}'.")

# Function to combine sentences until a chunk with at least 20 words is formed
def combine_sentences(sentences):
    combined_chunk = ''
    for sentence in sentences:
        if len(word_tokenize(sentence)) < 20:
            combined_chunk += sentence + ' '
        else:
            if combined_chunk:
                yield combined_chunk.strip()
                combined_chunk = ''
            yield sentence
    if combined_chunk:
        yield combined_chunk.strip()

def find_text_files(directory, ticker, filing_type):
    # List to store the paths of matching text files
    matching_files = []

    # Iterate over all files and directories in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a text file and if the file name contains the ticker and filing type
            if file.endswith(".txt") and ticker.upper() in file.upper() and filing_type.upper() in file.upper():
                # If found, append the full path to the list of matching files
                matching_files.append(os.path.join(root, file))

    return matching_files

def pipeline(ticker, companies_csv, directory_path, hybrid_search=True, semantic_search=True, filing_type="10-K", dense_encoder_model="paraphrase-mpnet-base-v2",device = "cpu", visualize_data=False):
    results_dir = f"./results/{ticker}"
    ensure_directory_exists(results_dir)

    required_files = []

    if filing_type == "10-K":
        matching_files = find_text_files(directory_path, ticker, filing_type)

        if matching_files:
            print("Matching text files found:")
            for file_path in matching_files:
                print(file_path)
            required_files.append(matching_files[-1])
        else:
            extract_company(ticker=ticker, all_companies_file=f"{os.getcwd()}/SEC-EDGAR-text/all_companies_list.txt")
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            command = f"python SEC-EDGAR-text --companies_list={os.getcwd()}/SEC-EDGAR-text/company_info.txt --storage={os.getcwd()}/SEC-EDGAR-text/sec_edgar_new --start={start_date} --end={end_date} --filings=10-K --report_period=all"  # Example command to list files in the current directory
            print(command)
            os.system(command)
            matching_files = find_text_files(directory_path, ticker, filing_type)
            if matching_files:
                print("Matching text files found:")
                for file_path in matching_files:
                    print(file_path)
                required_files.append(matching_files[-1])
            else:
                print(f"The Ticker({ticker}) is not listed in SEC EDGAR")
    
    else:
        matching_files_10_K = find_text_files(directory_path, ticker, "10-K")
        matching_files_10_Q = find_text_files(directory_path, ticker, "10-Q")

        if matching_files_10_K and matching_files_10_Q:
            print("Matching text files found:")
            for file_path in matching_files_10_K:
                print(file_path)
            for file_path in matching_files_10_Q:
                print(file_path)
            required_files.append(matching_files_10_K[-1])
            required_files.append(matching_files_10_Q[-1])
            required_files.append(matching_files_10_Q[-2])
            required_files.append(matching_files_10_Q[-3])
        else:
            extract_company(ticker=ticker, all_companies_file=f"{os.getcwd()}/SEC-EDGAR-text/all_companies_list.txt")
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            command = f"python SEC-EDGAR-text --companies_list={os.getcwd()}/SEC-EDGAR-text/company_info.txt --storage={os.getcwd()}/SEC-EDGAR-text/sec_edgar_new --start={start_date} --end={end_date} --filings=10-K,10-Q --report_period=all"  # Example command to list files in the current directory
            print(command)
            os.system(command)
            matching_files_10_K = find_text_files(directory_path, ticker, "10-K")
            matching_files_10_Q = find_text_files(directory_path, ticker, "10-Q")
            if matching_files_10_K and matching_files_10_Q:
                print("Matching text files found:")
                for file_path in matching_files_10_K:
                    print(file_path)
                for file_path in matching_files_10_Q:
                    print(file_path)
                required_files.append(matching_files_10_K[-1])
                required_files.append(matching_files_10_Q[-1])
                required_files.append(matching_files_10_Q[-2])
                required_files.append(matching_files_10_Q[-3])
            else:
                print(f"The Ticker {ticker} is not listed in the SEC EDGAR server.")
        
    # data = pd.read_csv(companies_csv)
    # imp_data = data[data["ticker"] == ticker]

    data = pd.read_csv(companies_csv)
    key_initiatives = list(data['Key Phrase'])

    # data = pd.read_csv(companies_csv)
    # data = data[data['Value Type'] == "Key Phrase"]
    # key_initiatives = list(data['value'])
    # product_lines = list(data['Product Line'])

    # data = pd.read_csv(companies_csv)
    # products = list(data.keys())[2:]
    # #print(products)
    # product_lines = []
    # key_initiatives = []
    # for product in products:
    #     sentences_list = [sentence.strip() for sentence in data[product][2].split(',')]
    #     for sentence in sentences_list:
    #         key_initiatives.append(sentence)
    #         product_lines.append(product)

    file_name = "combined_output.txt"

    try:
        with open(file_name, 'w') as outfile:
            for file_path in required_files:
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")  # Add a newline character to separate contents of each file
        print(f"All files have been successfully combined into {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    #file_name = matching_files[-1]

    sentence_model = SentenceTransformer(dense_encoder_model, device=device)
    # key_initiatives = list(imp_data[(imp_data["ticker"] == ticker) & (imp_data["value_type"] == "initiatives")]["value"])
    key_initiatives_embeddings = sentence_model.encode(key_initiatives)

    # Read the text from the file
    with open(file_name, 'r') as file:
        text = file.read()

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Combine sentences into chunks with at least 20 words
    chunks = list(combine_sentences(sentences))

    dense_scores = []
    for chunk in chunks:
        similarities_chunk = []
        chunk_embedding = sentence_model.encode(chunk)
        for i in range(len(key_initiatives)):
            #print(len(chunk_embedding), len(key_initiatives_embeddings[i]))
            similarity_score = cosine_similarity([chunk_embedding], [key_initiatives_embeddings[i]])[0][0]
            similarities_chunk.append(similarity_score)

        dense_scores.append(similarities_chunk)

    dense_scores = np.array(dense_scores)
    sparse_scores = np.zeros_like(dense_scores)

    tokenized_corpus = [doc.split(" ") for doc in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    for i in range(len(key_initiatives)):
        query = key_initiatives[i]
        tokenized_query = query.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)
        for j in range(len(doc_scores)):
            sparse_scores[j][i] = doc_scores[j]
    
     # Normalize the scores
    dense_scores_normalized = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
    sparse_scores_normalized = (sparse_scores - np.min(sparse_scores)) / (np.max(sparse_scores) - np.min(sparse_scores))
    
            
    if visualize_data:
        # Flatten the total_similarities list
        flat_similarities = np.array(dense_scores_normalized).flatten()

        # Visualize the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(flat_similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Dense Scores')
        plt.xlabel('Dense Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{results_dir}/{ticker}_{filing_type}_dense_score_distribution.png")
        #plt.show()

        # Box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot(flat_similarities, vert=False)
        plt.title('Box Plot of Dense Scores')
        plt.xlabel('Dense Score')
        plt.grid(True)
        plt.savefig(f"{results_dir}/{ticker}_{filing_type}_dense_score_box_plot.png")
        #plt.show()

        # Flatten the total_similarities list
        flat_similarities = np.array(sparse_scores_normalized).flatten()

        # Visualize the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(flat_similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Sparse Scores')
        plt.xlabel('Sparse Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{results_dir}/{ticker}_{filing_type}_sparse_score_distribution.png")
        #plt.show()

        # Box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot(flat_similarities, vert=False)
        plt.title('Box Plot of Sparse Scores')
        plt.xlabel('Sparse Score')
        plt.grid(True)
        plt.savefig(f"{results_dir}/{ticker}_{filing_type}_sparse_score_box_plot.png")
        #plt.show()
    if hybrid_search:
        dense_scores_counter = 0
        sparse_scores_counter = 0
        # Flatten the total_similarities list
        flat_similarities = np.array(dense_scores_normalized).flatten()

        # Calculate statistics
        mean_similarity = np.mean(flat_similarities)
        std_dev_similarity = np.std(flat_similarities)
        dense_threshold = mean_similarity + 4*std_dev_similarity

        # Flatten the total_similarities list
        flat_similarities = np.array(sparse_scores_normalized).flatten()

        # Calculate statistics
        mean_similarity = np.mean(flat_similarities)
        std_dev_similarity = np.std(flat_similarities)
        sparse_threshold = mean_similarity + 2*std_dev_similarity

        initiative_counter = {}
        with open(f'{results_dir}/{ticker}_{filing_type}_hybrid_search_key_initiatives.txt', 'w') as file:
            file.write(f"Here are the context of the key initiatives in the {ticker}'s {filing_type} filing\n")
        
        for i in range(len(key_initiatives)):
            #print(f"Key Initiative: {key_initiatives[i]}")
            with open(f'{results_dir}/{ticker}_{filing_type}_hybrid_search_key_initiatives.txt', 'a') as file:
                # file.write(f"Key Initiative: {key_initiatives[i]} \n\n")
                file.write(f"Key Initiative: {key_initiatives[i]} \nProduct Line: {str(data['Product Line'][i])}\n\n")
                # file.write(f"Key Initiative: {key_initiatives[i]} \nProduct Line: {str(product_lines[i])}\n\n")
            chunks_count = 0
            for j in range(len(chunks)):
                if dense_scores_normalized[j][i] >= dense_threshold:
                    dense_scores_counter += 1
                if sparse_scores_normalized[j][i] >= sparse_threshold:
                    sparse_scores_counter += 1
                if dense_scores_normalized[j][i] >= dense_threshold and sparse_scores_normalized[j][i] >= sparse_threshold:
                    chunks_count += 1
                    with open(f'{results_dir}/{ticker}_{filing_type}_hybrid_search_key_initiatives.txt', 'a') as file:
                        file.write(f"{chunks[j]}\n\n")
            
            #print(f"Chunks Count: {chunks_count}")
            initiative_counter[key_initiatives[i]] = chunks_count

        #print(dense_scores_counter)
        #print(sparse_scores_counter)
        with open(f'{results_dir}/{ticker}_{filing_type}_hybrid_search_key_initiatives.txt', 'a') as file:
            file.write("\n\n\n\n\n\nKey Initiatives Counter: \n")
        counter_keys = list(initiative_counter.keys())
        for key in counter_keys:
            with open(f'{results_dir}/{ticker}_{filing_type}_hybrid_search_key_initiatives.txt', 'a') as file:
                file.write(f"{key} : {initiative_counter[key]}\n\n")
    if semantic_search:
        dense_scores_counter = 0
        # Flatten the total_similarities list
        flat_similarities = np.array(dense_scores_normalized).flatten()

        # Calculate statistics
        mean_similarity = np.mean(flat_similarities)
        std_dev_similarity = np.std(flat_similarities)
        aggression_scale = 40
        dense_threshold = mean_similarity + (aggression_scale*std_dev_similarity/10)

        initiative_counter = {}
        with open(f'{results_dir}/{ticker}_{filing_type}_key_initiatives_{aggression_scale}.txt', 'w') as file:
            file.write(f"Here are the context of the key initiatives in the {ticker}'s {filing_type} filing\n")
        
        for i in range(len(key_initiatives)):
            #print(f"Key Initiative: {key_initiatives[i]}")
            with open(f'{results_dir}/{ticker}_{filing_type}_key_initiatives_{aggression_scale}.txt', 'a') as file:
                #file.write(f"Key Initiative: {key_initiatives[i]} \n\n")
                file.write(f"Key Initiative: {key_initiatives[i]} \nProduct Line: {str(data['Product Line'][i])}\n\n")
                # file.write(f"Key Initiative: {key_initiatives[i]} \nProduct Line: {str(product_lines[i])}\n\n")
            chunks_count = 0
            for j in range(len(chunks)):
                if dense_scores_normalized[j][i] >= dense_threshold:
                    dense_scores_counter += 1
                if dense_scores_normalized[j][i] >= dense_threshold:
                    chunks_count += 1
                    with open(f'{results_dir}/{ticker}_{filing_type}_key_initiatives_{aggression_scale}.txt', 'a') as file:
                        file.write(f"\t\t{chunks[j]}\n\n")
            
            #print(f"Chunks Count: {chunks_count}")
            initiative_counter[key_initiatives[i]] = chunks_count

        #print(dense_scores_counter)
        #print(sparse_scores_counter)
        with open(f'{results_dir}/{ticker}_{filing_type}_key_initiatives_{aggression_scale}.txt', 'a') as file:
            file.write("\n\n\nKey Initiatives Counter: \n")
        counter_keys = list(initiative_counter.keys())
        for key in counter_keys:
            with open(f'{results_dir}/{ticker}_{filing_type}_key_initiatives_{aggression_scale}.txt', 'a') as file:
                file.write(f"{key} : {initiative_counter[key]}\n\n")

if __name__ == '__main__':
    tickers = ["AIG", "CSCO"]
    for ticker in tickers:
        pipeline(ticker=ticker, companies_csv="./data/whatfix_poc_-_key_phrases.csv", directory_path="./SEC-EDGAR-text", hybrid_search=True, semantic_search=True, filing_type="all", visualize_data=True)
