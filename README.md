# A Serverless RAG Chatbot Framework using AWS OpenSearch, AWS Bedrock and LangChain
This is a RAG chatbot application based on AWS components and designed to be optimized for a serverless architecture
and cost optimized for a high-volume mobile application use case.  The AWS Services being used are:
- AWS Bedrock with the Titan LLM and Titan embeddings 
- OpenSearch (aka Elasticsearch) vector database - this is instance based but can be serverless (not implemented yet)

The frameworks I created here abstract out a variety of components to enable easily testing variations.  This makes it easier to tune the implementation based on the 
content being use in my applications.  The goal is to play with different LLM's, different embeddings, parameters like temperature, top_p and more.
Content loading also needed to be highly refined allowing me to control each source easily.

This library is meant more to provide a complete working set of examples and less to be an out-of-the-box library to use as-is.  
Because this framework imports so many variants of embeddings, LLM models, etc. it is not suitable for a direct deployment into something like AWS Lambda... it is a :pig:.  


## Features and aspects of interest
- **LangChain** - Great framework building Generative AI applications. 
- **LLM Model Support** - Defaults to AWS Bedrock Titan LLM, but supports other Bedrock models (LLama 2, Jurassic, Anthropic Claude) OpenAI (GTP-3.5 and GTP-4), Google Gemini
- **LangChain LLM Callbacks** - Example of using LLM Callbacks, provided here for custom costing 
- **Conversational Memory** -- Designed to manage chat history memory from the client side - to support a serverless model
- **OpenSearch** - an AWS hosted semantic search service which has a vector database that is valuable in the retrieval feature of RAG
- **OpenSearch loading library** - making it easier to load multiple content sources into an OpenSearch vector database.
- **Langchain LCEL Chains** - for more flexible chaining of steps in the RAG app
- **Multiple embedding models** - Defaults to Bedrock Titan, OpenAI and Hugging Face and Coherence
- **Web and directory crawlers** - for loading content into the vector DB.  Lots of fine-tuning to document selection features - whitelisting, black listing, etc.
- **Prompt library management** - making it easier to implement query routing to optimize for specific domains of questions
- **LangSmith logging** - for logging and debugging

## How to use
### Prerequisites
- Python 3.11 (may work on older, but this was the target version used)
- AWS Account with keys defined in .aws/credentials file
- AWS OpenSearch Domain created and accessible to the AWS account credentials above
- AWS Bedrock with Titan Express LLM (or other LLM supported by this code) - you may need to request access
- Ideally a LangChain (LangSmith) API Key defined in .env file (or removed if not using LangSmith) for logging and monitoring
- If using google gemini or openai, you will need to have an account and keys for those services and have them defined in the .env file
- Python dependencies installed (dependency manager for this project coming soon)

### Setting up the project locally
This project uses Poetry to manage dependencies.  You can install it with the following command:
```bash
pip install poetry
```

Then you can install the dependencies with the following command:
```bash
poetry install
```

Alternatively, you can install the package from PyPi with the following command:
```bash
pip install aws-rag-bot
```
Where the PyPi package is available at https://pypi.org/project/aws-rag-bot/
****
### Sample Code
This is a very simple, high-level example.  Check out the rag_bot_code_samples.ipynb for a more.
First step is to have content in your vector database.  
```python
from open_search_vector_db.aws_opensearch_vector_database import OpenSearchVectorDBLoader
content_sources = [{"name": "Internal KB Docs", "type": "PDF", "location": "kb-docs"}]
vectordb_loader = OpenSearchVectorDBLoader(domain_name=my_open_search_domain_name,  
                                     index_name=my_index_name,
                                     data_sources=content_sources)

vectordb_loader.load()
```

Then you can start asking questions of it
```python
from rag_chatbot import RagChatbot, LlmModelTypes
from prompt_library import DefaultPrompts
chatbot = RagChatbot(my_open_search_domain_name,
                     LlmModelTypes.BEDROCK_TITAN_EXPRESS,
                     prompt_model=NasaSpokespersonPrompts)
chat_history = []
question = "What...?" # Ask a question related to the content you loaded

response = chatbot.ask_question(question, chat_history, verbose=True)
print(response)
```


### Provisioning a test index in OpenSearch
You can use the tests/provision_test_index.py to create a test index in OpenSearch.  The content loaded
supports the test cases in this project


### Running chatbot_client.py
A very simple command line client program has been created as an example and tool to test.  It is called chatbot_client.py.  
It is a simple command line program that will ask a question and then print the response while retaining the chat history for context.  

```bash
python chatbot_client.py my-opensource-domain-name
```

### Running tests
There are two test modules in the tests folder used to run through search and the RAG bot to make sure everything is working as well as provide
some additional samples of how to use the framework.

### Using Ragas evaluation framework
Ragas is one of a variety of evaluation tools for RAG applications.  
It can be used to evaluate both the retrieval and generation aspects of the RAG bot.  
With an evaluation tool you can then use this project's features to vary whatever aspects you need
and compare the results to make decisions and tune. 
A simple example of this can be found in the tests folder.


## References
**Vector Database:**  May references show using Chroma and FAISS, but I needed a solution that worked well in a Lambda serverless environment.  
Ideally it would be at AWS keeping my stack uniform.  
- https://aws.amazon.com/what-is/vector-databases/

Ultimately I chose OpenSearch because of cost, support by LangChain and a serverless version I plan to evaluate

**Vector Database Loader:**  LangChain has a great library of DataLoaders for loading data into OpenSearch.  I wanted an effective way
to scrape a website with help from this article chose to use Selenium.
- https://www.comet.com/site/blog/langchain-document-loaders-for-web-data/

I also used the directory loader and plan to implement cloud based directory loaders in the future.  Primarily S3 and Google Drive.

**Langsmith Logging and Debugging:**  I used LangSmith for logging and debugging.  It is a great tool for this purpose.
- https://docs.smith.langchain.com


### What's next (Roadmap):
1. Add vector database stub for FAISS and easy testing
1. Expand range of support for Embeddings - including Hugging Face cloud service for embeddings
