# Talent Scan AI

Talent Scan AI is a RAG (Retrieval-Augmented Generation) pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. The system uses a similarity retriever to identify the most suitable applicants based on job descriptions, and then augments this data into an LLM generator for downstream tasks such as analysis, summarization, and decision-making.

## System Architecture

[![](https://mermaid.ink/img/pako:eNqVVNFumzAU_RXLe6UdhDQQKk2iARKaVsuStg8jfbDgJkE1kBnTJSv99xkDGbQPU0BCtu-5h-N7rv2GwywCbOEtI_sdenDWKRKPHTzmwJ7RxcW38kcB7Pj1cU8zEpXoJlhxBiShMUd-yoFtSAjPddaNxNsLH83hWKJJ8H0Pqe2jauWJ0DgiPM7SHnjheMiLKeQlcoIl5EUCaMGyEPI8Yw3SkUj3wBkJOUToAQ68RG7gEE6Ehi3kHVpXgp0sFEwpR5Ndkb4Ici_wbH-1Qk8Q8oyhlfi0qrtyql0jueESTQM5-CRnWguvV4WcBj4LlvYULYGzGF6hBXsS3K5GqBUmJM1qxKxBUHglQnBdAxH2g7u7e2RvBbgndCLxTTkFoyy1X8f8kwFL-FWIupTotjVhunhA98Jr2rDdSuwUUmCSR_x4n6U5nMi6lF7GEsL7sHnQjlHDcirRXCbVjSJaZp3WyyElee7ABhWizKfmQZuYUuuLaUxcz1ZyzrIXsL7out6ML37HEd9Zg_3h-gONKABpfIjTbcMzVl13rJ7F8yqbQvZEQ-J5jqGeR8Jqjwk9Udiqcd5-KE0-bcdxbNVxzqIRxwNYSmjVB40W1R2Zk7NIKotaCXr1_i-7k49smd6lRDfKvG97LzpRbruyezFHcT843Qt7Xft6kaky--dKL-L3S32NFZyA6PA4EjfhW4VcY76DBNbYEsMINqSgfI3X6buAkoJnq2MaYouzAhTMsmK7w9aG0FzMin11Lp2YiBs1aSF7kv7Msu4UW2_4gK3h6FLTdG00UDVVH5n6SMFHbJnG5dC8MnTTGOiGOdSG7wr-I_O1y-HVWB2ahq5fmWNd1wbvfwEYwt1Z?type=png)](https://mermaid.live/edit#pako:eNqVVNFumzAU_RXLe6UdhDQQKk2iARKaVsuStg8jfbDgJkE1kBnTJSv99xkDGbQPU0BCtu-5h-N7rv2GwywCbOEtI_sdenDWKRKPHTzmwJ7RxcW38kcB7Pj1cU8zEpXoJlhxBiShMUd-yoFtSAjPddaNxNsLH83hWKJJ8H0Pqe2jauWJ0DgiPM7SHnjheMiLKeQlcoIl5EUCaMGyEPI8Yw3SkUj3wBkJOUToAQ68RG7gEE6Ehi3kHVpXgp0sFEwpR5Ndkb4Ici_wbH-1Qk8Q8oyhlfi0qrtyql0jueESTQM5-CRnWguvV4WcBj4LlvYULYGzGF6hBXsS3K5GqBUmJM1qxKxBUHglQnBdAxH2g7u7e2RvBbgndCLxTTkFoyy1X8f8kwFL-FWIupTotjVhunhA98Jr2rDdSuwUUmCSR_x4n6U5nMi6lF7GEsL7sHnQjlHDcirRXCbVjSJaZp3WyyElee7ABhWizKfmQZuYUuuLaUxcz1ZyzrIXsL7out6ML37HEd9Zg_3h-gONKABpfIjTbcMzVl13rJ7F8yqbQvZEQ-J5jqGeR8Jqjwk9Udiqcd5-KE0-bcdxbNVxzqIRxwNYSmjVB40W1R2Zk7NIKotaCXr1_i-7k49smd6lRDfKvG97LzpRbruyezFHcT843Qt7Xft6kaky--dKL-L3S32NFZyA6PA4EjfhW4VcY76DBNbYEsMINqSgfI3X6buAkoJnq2MaYouzAhTMsmK7w9aG0FzMin11Lp2YiBs1aSF7kv7Msu4UW2_4gK3h6FLTdG00UDVVH5n6SMFHbJnG5dC8MnTTGOiGOdSG7wr-I_O1y-HVWB2ahq5fmWNd1wbvfwEYwt1Z)

## Components

1. **main.py**: The entry point of the application. It sets up the Streamlit interface and manages the overall flow of the application.

2. **llm_agent.py**: Contains the ChatBot class, which is responsible for generating responses using the OpenAI API.

3. **data_ingestor.py**: Handles the ingestion of resume data into the vector store.

4. **retriever.py**: Implements the RAG retriever, which is responsible for finding relevant resumes based on the user's query.

5. **chatbot_verbosity.py**: Handles the verbosity of the chatbot's responses.

## Key Features

- PDF resume upload and processing
- OpenAI API integration for natural language processing
- Vector store for efficient similarity search
- Streamlit-based user interface for easy interaction
- Logging for system monitoring and debugging

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key in the Streamlit interface.

3. Run the application:
   ```
   cd app
   streamlit run main.py
   ```

4. Upload resume PDFs and start querying the system with job descriptions or specific questions about candidates.

## Usage

1. Add your OpenAI API key in the sidebar.
2. Upload resume PDFs using the file uploader in the sidebar.
3. Type your job description or query in the chat input.
4. The system will process your query, retrieve relevant resumes, and generate a response.

## Configuration

You can configure the following settings in the sidebar:
- OpenAI API Key
- GPT Model (default: gpt-3.5-turbo)
- Upload resumes (PDF format)


## Note

This system is designed to assist hiring managers in their decision-making process. It should be used as a tool to complement human judgment, not replace it entirely.