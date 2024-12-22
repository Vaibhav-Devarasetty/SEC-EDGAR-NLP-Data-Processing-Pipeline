from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from datetime import datetime, timedelta
import os
import sys

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

# key_phrases = [
    #     "Targeted sales team training: Customizable training programs focused on specific skills and needs of different sales teams.",
    #     "Skills development for sales: Continuous enhancement of sales skills through interactive and engaging modules.",
    #     "Continuous learning for sales: Ongoing educational initiatives to keep sales teams updated with the latest trends and techniques.",
    #     "Performance optimization training: Training aimed at improving the overall performance and effectiveness of sales representatives.",
    #     "Enabling sales team success: Providing tools and resources to ensure sales teams achieve their targets and objectives.",
    #     "Centralized content solution: A single platform for storing, managing, and accessing all sales-related content.",
    #     "Streamline sales materials: Organizing and simplifying access to sales documents and resources.",
    #     "Sales content accessibility: Ensuring all sales materials are easily accessible to the sales team.",
    #     "Improved content control: Better management and control over the distribution and use of sales content.",
    #     "Efficient content management: Streamlined processes for managing and updating sales content.",
    #     "Data-driven sales coaching: Utilizing data to provide personalized coaching and feedback to sales reps.",
    #     "Real-time coaching insights: Immediate feedback and insights during or after sales interactions.",
    #     "Boosting seller effectiveness: Enhancing the skills and performance of individual sellers through targeted initiatives.",
    #     "Personalized sales mentorship: One-on-one mentoring tailored to the specific needs of each sales rep.",
    #     "Sales performance improvement: Programs and tools designed to continuously improve sales performance.",
    #     "Sales force preparedness: Ensuring sales teams are well-prepared for all potential scenarios.",
    #     "Engagement readiness assessment: Evaluating how ready the sales team is to engage with prospects effectively.",
    #     "Readiness metrics tracking: Monitoring key metrics to assess the readiness of the sales force.",
    #     "Sales readiness insights: Gaining insights into the preparedness of the sales team.",
    #     "Preparedness analytics: Analyzing data to understand and improve sales readiness.",
    #     "AI-driven sales insights: Leveraging AI to gain deeper insights into sales activities and performance.",
    #     "Analyzing sales conversations: Reviewing and analyzing sales calls to identify areas for improvement.",
    #     "Sales call intelligence: Tools to understand and improve the effectiveness of sales calls.",
    #     "Improve conversation tactics: Enhancing the techniques used by sales reps during conversations with prospects.",
    #     "Enhanced customer engagement: Strategies to increase and improve interactions with customers.",
    #     "Sales performance visualization: Visual tools to track and understand sales performance metrics.",
    #     "Informed decision-making data: Data that supports better decision-making in sales strategies.",
    #     "Dashboard sales KPI tracking: Dashboards that track key performance indicators for sales.",
    #     "Sales metrics analysis: Analyzing sales data to identify trends and areas for improvement.",
    #     "Strategic sales planning: Developing long-term strategies to achieve sales goals.",
    #     "Personalized sales interaction: Tailoring sales interactions to the specific needs and preferences of each customer.",
    #     "Secure buyer-seller space: Providing a secure environment for buyers and sellers to interact.",
    #     "Digital content sharing room: A digital platform for sharing sales content with prospects.",
    #     "Custom sales room experience: Personalized virtual rooms for engaging with prospects.",
    #     "Interactive deal rooms: Collaborative spaces for negotiating and finalizing deals.",
    #     "Predictive sales analytics: Using data analytics to predict future sales trends and outcomes.",
    #     "Sales ROI forecasting: Estimating the return on investment for sales activities.",
    #     "Accurate revenue insights: Providing precise insights into revenue generation.",
    #     "Investment return analysis: Analyzing the returns on sales investments.",
    #     "Strategic forecasting insights: Gaining insights to support strategic planning and forecasting."
    # ]

key_phrases =  ["Mindtickle's sales enablement platform offers comprehensive training programs designed to improve sales performance. It provides tools for onboarding, continuous learning, and skill development, enabling sales teams to stay updated and effective. The platform uses personalized learning paths, interactive content, and analytics to track progress and ensure sales readiness.", 
                "Mindtickle's Sales Content Management Solutions provide a centralized platform for managing and distributing sales content. The service enables organizations to store, organize, and share sales materials such as presentations, case studies, and brochures. It ensures that sales teams have easy access to the most up-to-date content, supports content personalization, and tracks content usage and effectiveness. This helps improve sales productivity, consistency in messaging, and overall sales performance.",
                "Mindtickle's Sales Coaching Software is designed to enhance sales team performance through structured coaching programs. It offers features such as skill assessments, personalized learning paths, feedback mechanisms, and analytics to monitor progress. This software helps managers deliver targeted coaching, track improvements, and ensure that sales representatives are well-prepared and continuously developing their skills.",
                "Mindtickle's Sales Readiness Index analyzes sales team performance to ensure sales readiness. The platform provides insights into the effectiveness of sales training, coaching, and content usage. It helps organizations measure skill levels, identify knowledge gaps, and track improvements over time. This data-driven approach enables sales leaders to make informed decisions and optimize their team's performance for better results.",
                "Mindtickle's Conversation Intelligence platform analyzes sales calls to provide insights for improving sales performance. It uses AI to transcribe and evaluate conversations, identifying key trends, customer sentiments, and areas for improvement. This helps sales teams refine their messaging, understand customer needs better, and enhance overall sales effectiveness.",
                "Mindtickle's Revenue Intelligence platform provides insights into sales performance and revenue trends. It integrates data from various sources to offer a comprehensive view of sales activities, pipeline health, and forecast accuracy. This helps sales leaders make informed decisions, identify opportunities, and address challenges to drive revenue growth.",
                "Mindtickle's Digital Sales Rooms offer a personalized virtual space for sales teams to engage with prospects and customers. These rooms provide a centralized location for sharing content, tracking engagement, and facilitating communication throughout the sales process. This tool aims to enhance the buyer's experience, streamline interactions, and improve the effectiveness of sales engagements.",
                "Mindtickle's Sales Enablement Analytics Software provides insights into the effectiveness of sales training and content. It tracks engagement, completion rates, and performance metrics to help sales leaders understand the impact of their enablement programs. The platform uses data-driven analysis to identify areas for improvement, optimize training strategies, and ensure that sales teams are well-prepared to meet their targets."]


def pipeline(company_name, ticker, key_phrases, directory_path="./SEC-EDGAR-text"):

    client = OpenAI(api_key='')
    company_name = company_name
    instruction_prompt = f"""
    You are an expert data analyst. Your task is to review the text files meticulously to find matches of strategies and initiatives that align with the product descriptions of {company_name} as mentioned in the messages. For each match found, provide the following:

    1. Provide a quote directly from the texts where the match was found
    2. Explain why the quote is relevant to the product description.
    3. Include citations for the quotes from the respective text files.

    Ensure the quotes are accurate, directly related to the value propositions, justified by it's explaination and properly cited.
    """
    
    assistant = client.beta.assistants.create(
    name="Company Filings Analyser",
    instructions=instruction_prompt,
    model="gpt-4o",
    tools=[{"type": "file_search"}]
    )

    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(name="Company Filings")

    ticker = ticker
    directory_path=directory_path

    results_dir = f"./results/{ticker}"
    ensure_directory_exists(results_dir)

    required_files = []

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

    # Ready the files for upload to OpenAI
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

    file_paths = required_files
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    key_phrases = key_phrases

    messages = []

    for key_phrase in key_phrases:
        content = f"""
            The Product Description is "{key_phrase}". 
            1. Provide a quote directly from the text files where the match was found
            2. Explain why the quote is relevant to the product description.
            3. Include citations for the quotes from the respective text files.
        """

        messages.append({
        "role": "user",
        "content": content
        })

    messages_list = []

    max_messages = 10

    for i in range(int(len(messages)/max_messages)):
        messages_list.append(messages[i*max_messages:(i+1)*max_messages])

    messages_list.append(messages[(int(len(messages)/max_messages))*max_messages:])


    for i in range(len(messages_list)):

        thread = client.beta.threads.create(
            messages=messages_list[i]
        )
        
        # The thread now has a vector store with that file in its tool resources.
        print(thread.tool_resources.file_search)

        # output_file_path = f'{results_dir}/{company_name}_{ticker}_GPT_file_search.txt'
        
        class EventHandler(AssistantEventHandler):
            @override
            def on_text_created(self, text) -> None:
                print(f"\nassistant > ", end="", flush=True)

            @override
            def on_tool_call_created(self, tool_call):
                print(f"\nassistant > {tool_call.type}\n", flush=True)

            @override
            def on_message_done(self, message) -> None:
                # print a citation to the file searched
                message_content = message.content[0].text
                annotations = message_content.annotations
                citations = []
                for index, annotation in enumerate(annotations):
                    message_content.value = message_content.value.replace(
                        annotation.text, f"[{index}]"
                    )
                    if file_citation := getattr(annotation, "file_citation", None):
                        cited_file = client.files.retrieve(file_citation.file_id)
                        citations.append(f"[{index}] {cited_file.filename}")

                print(message_content.value)
                print("\n".join(citations))


        # Then, we use the stream SDK helper
        # with the EventHandler class to create the Run
        # and stream the response.

        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()

            # Your code here
            #print("This text will be saved to the file.")


if __name__ == '__main__':
    tickers = ["CSCO", "AIG"]
    company_name = "fornow"
    for ticker in tickers:
        results_dir = f"./results/{ticker}"
        ensure_directory_exists(results_dir)
        output_file_path = f'{results_dir}/{company_name}_{ticker}_GPT_file_search.txt'
        with open(output_file_path, "w") as f:
            # Store the reference to the original standard output
            original_stdout = sys.stdout

            # Redirect the standard output to the file
            sys.stdout = f
            
            # Your main script code here
            pipeline(company_name=company_name, ticker=ticker, key_phrases=key_phrases)

            # Restore the original standard output
            sys.stdout = original_stdout