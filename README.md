# Project Title: SEC-EDGAR Analysis and Search Pipelines

## Abstract Overview

This project involves a series of Python scripts designed to process and analyze data from SEC-EDGAR filings. The main objective is to extract and classify key information using natural language processing (NLP) techniques. The project leverages advanced models and methods, including GPT-3, to search, summarize, and classify textual data. The architecture follows a pipeline model where each script handles a specific task in the overall workflow.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Usage](#usage)
   - [Running the Complete Pipeline](#running-the-complete-pipeline)
   - [Description of Scripts](#description-of-scripts)
3. [Models and Methods](#models-and-methods)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Contributing](#contributing)
6. [License](#license)

## Project Structure

The project directory contains the following files:

- `domain_specific.py`
- `high_prob_chunks.py`
- `langchain_search.py`
- `summary_based_agent_search.py`
- `summary_chunks_classify.py`
- `complete_pipeline.py`
- `gpt_file_search.py`

## Usage

### Running the Complete Pipeline

To run the complete pipeline, execute the `complete_pipeline.py` script:

```sh
python complete_pipeline.py
```

### Description of Scripts

#### `domain_specific.py`
Handles domain-specific preprocessing tasks, such as filtering and extracting relevant sections from the raw text data.

#### `high_prob_chunks.py`
Identifies high-probability chunks of text that are likely to contain key information based on predefined criteria.

#### `langchain_search.py`
Implements a search mechanism using LangChain to retrieve relevant documents or text snippets.

#### `summary_based_agent_search.py`
Performs a summary-based search using an agent-based approach to refine the search results.

#### `summary_chunks_classify.py`
Classifies summarized text chunks into various categories based on their content and relevance.

#### `complete_pipeline.py`
Integrates all the individual scripts into a cohesive pipeline, orchestrating the entire process from raw data ingestion to final output generation.

#### `gpt_file_search.py`
Utilizes OpenAI's GPT-3 to search and extract information from large text files, leveraging the model's language understanding capabilities.

## Models and Methods

- **Natural Language Processing (NLP):** Various NLP techniques are used to process and analyze text data.
- **GPT-3:** Employed for its advanced language understanding and generation capabilities.
- **LangChain:** Used for document retrieval and search functionalities.
- **Classification Algorithms:** Applied to categorize and classify text chunks based on their content.

## Pipeline Architecture

1. **Data Ingestion:** Raw data is ingested from SEC-EDGAR filings.
2. **Preprocessing:** Domain-specific preprocessing is performed to clean and filter the data.
3. **Chunk Identification:** High-probability chunks of text are identified for further analysis.
4. **Search and Retrieval:** Relevant documents or text snippets are retrieved using LangChain and GPT-3.
5. **Summarization:** Summaries are generated to condense the information.
6. **Classification:** Summarized chunks are classified into relevant categories.
7. **Output Generation:** The final output is generated and saved in the specified format.
