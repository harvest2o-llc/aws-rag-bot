from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    SeleniumURLLoader,
    PyPDFLoader
)
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
import requests
import xml.etree.ElementTree as ET

from open_search_vector_db.kustomer import get_all_shortcuts_as_documents
import re

DEFAULT_CHUNK_SIZE = 256


# Defined structures for the supported embedding types.  Can be extended
class EmbeddingTypes:
    OPENAI_GPT_DEFAULT = {"name": "openai-gpt-default", "provider": "openai", "model": "default"}
    HUGGING_FACE_DEFAULT = {"name": "hugging-face-default", "provider": "hugging-face", "model": "default"}
    BEDROCK_DEFAULT = {"name": "bedrock-default", "provider": "bedrock", "model": "amazon.titan-embed-text-v1",
                       "region": "us-east-1"}


# ======= Static Utility Functions =======
def clean_documents_newlines_spaces_and_tabs(docs):
    # This will remove all the redundant newlines, spaces, and tabs from the documents
    #  with a good document spitter, this may not be needed, but it does reduce character count
    #     and therefore is at a minimum and small optimization
    # Compile regular expressions for matching sequences of newlines, spaces, and tabs
    newline_re = re.compile(r'\n+')
    space_re = re.compile(r' {2,}')  # Matches two or more spaces
    tab_re = re.compile(r'\t+')  # Matches any sequence of tabs

    for doc in docs:
        # Clean the document content
        cleaned_content = newline_re.sub('\n', doc.page_content)
        cleaned_content = space_re.sub(' ', cleaned_content)
        cleaned_content = tab_re.sub(' ', cleaned_content)

        # Update the document's page_content in place
        doc.page_content = cleaned_content

    return docs


def get_opensearch_endpoint(domain_name, region=None):
    # Get the callable endpoint for the OpenSearch domain
    # Note that this can be used to make Elastic Search like calls directly
    client = None
    if region:
        client = boto3.client('es', region)
    else:
        client = boto3.client('es')

    response = client.describe_elasticsearch_domain(
        DomainName=domain_name
    )

    return response['DomainStatus']['Endpoint']


def get_embeddings_from_model(embedding_model=None):
    embeddings = None
    if embedding_model is None:
        embedding_model = EmbeddingTypes.BEDROCK_DEFAULT

    if embedding_model['name'] == EmbeddingTypes.OPENAI_GPT_DEFAULT['name']:
        load_dotenv(find_dotenv())
        embeddings = OpenAIEmbeddings()

    elif embedding_model['name'] == EmbeddingTypes.HUGGING_FACE_DEFAULT['name']:
        embeddings = HuggingFaceEmbeddings()

    elif embedding_model['name'] == EmbeddingTypes.BEDROCK_DEFAULT['name']:
        embeddings = BedrockEmbeddings(model_id=embedding_model['model'], region_name=embedding_model['region'])

    return embeddings


def get_urls_from_sitemap(sitemap_url):
    """Fetches and extracts all URLs from a given sitemap or sitemap index."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'application/xml;q=0.9, */*;q=0.8'
    }
    if 'library' in sitemap_url:  # Hack for the wordpress site.  may work for others, but it had problems
        response = requests.get(sitemap_url, headers=headers, timeout=60)
    else:
        response = requests.get(sitemap_url)

    response.raise_for_status()

    root = ET.fromstring(response.content)

    urls = []

    # Check if it's a sitemap index
    if root.tag.endswith('sitemapindex'):
        for sitemap in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
            loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            urls.extend(get_urls_from_sitemap(loc))
    # Else assume it's a regular sitemap
    else:
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            urls.append(loc)

    return urls


def scrape_website_pages(website_urls, headless=True):
    # Uses the Selenium engine to scrape the website pages and return Langchain documents
    # https://scrapeops.io/selenium-web-scraping-playbook/python-selenium-disable-image-loading/
    loader = SeleniumURLLoader(urls=website_urls, headless=headless,
                               arguments=["enable-features=NetworkServiceInProcess"])
    docs = loader.load()

    ready_to_use_docs = []

    for doc in docs:
        ready_to_use_docs.append(doc)
    return ready_to_use_docs


def get_documents_from_folder(content_source):
    # Read a folder of documents and return Langchain documents
    #  Uses content source for better control
    #  Handles a variety of document types
    # first make sure it is valid for me to handle
    search_expression = None
    recursive = True

    directory = content_source['location']
    if content_source['type'] == 'Microsoft Word':
        search_expression = "*.docx"
        loader_class = Docx2txtLoader

    elif content_source['type'] == 'PDF':
        # NOTE: The PDF parser will break the document up into pages, so one doc could translate into many
        search_expression = "*.pdf"
        loader_class = PyPDFLoader

    else:
        raise (f"   Error loading documents. Cannot handle loading documents of type {content_source['type']}")

    if 'recursive' in content_source:
        recursive = content_source['recursive']

    print(f"   Reading {content_source['type']} documents from {content_source['location']}")
    loader = DirectoryLoader(path=directory, glob=search_expression, loader_cls=loader_class, recursive=recursive)
    folder_docs = clean_documents_newlines_spaces_and_tabs(loader.load())

    print(f"   Documents loaded in memory.  Total count={len(folder_docs)}")

    filtered_docs = folder_docs
    if 'whitelist' in content_source:
        filtered_docs = filter_urls_with_whitelist(filtered_docs, content_source['whitelist'])

    if 'blacklist' in content_source:
        filtered_docs = filter_urls_with_blacklist(filtered_docs, content_source['blacklist'])

    print(
        f"   Found {len(folder_docs)} URLs. Filted down to {len(filtered_docs)}. Lets go screen scraping (the slow part)")

    if 'chunk_size' in content_source and content_source['chunk_size'] == 0:
        return filtered_docs
    else:
        doc_chunks = chunk_up_documents(filtered_docs,
                                            content_source['chunk_size'] if 'chunk_size' in content_source else None)
        return doc_chunks


def filter_urls_with_blacklist(urls, blacklist):
    # Separate exact matches from wildcard matches
    exact_matches = [url for url in blacklist if "*" not in url]
    wildcard_matches = [url.replace("*", ".*") for url in blacklist if "*" in url]

    # Compile regex patterns for wildcard matches
    wildcard_patterns = [re.compile(pattern) for pattern in wildcard_matches]

    # Filter URLs
    filtered_urls = []
    for url in urls:
        # Check against exact matches
        if url in exact_matches:
            continue

        # Check against wildcard patterns
        if any(pattern.match(url) for pattern in wildcard_patterns):
            continue

        # If URL does not match any blacklist entry, add it to the filtered list
        filtered_urls.append(url)

    return filtered_urls


def filter_urls_with_whitelist(urls, whitelist):
    # Separate exact matches from wildcard matches
    exact_matches = [url for url in whitelist if "*" not in url]
    wildcard_matches = [url.replace("*", ".*") for url in whitelist if "*" in url]

    # Compile regex patterns for wildcard matches
    wildcard_patterns = [re.compile(pattern) for pattern in wildcard_matches]

    # Filter URLs
    filtered_urls = []
    for url in urls:
        # Check against exact matches
        if url in exact_matches:
            filtered_urls.append(url)
            continue

        # Check against wildcard patterns
        if any(pattern.match(url) for pattern in wildcard_patterns):
            filtered_urls.append(url)
            continue

    return filtered_urls


def chunk_up_documents(documents, chunk_size):
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if chunk_size == 1:
        return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=26,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"   Chunked {len(documents)} documents into {len(docs_chunks)} chunks using size={chunk_size}")

    return docs_chunks


def get_documents_from_website(content_source, headless=True):
    # use the source definition to get documents form a website and return Langchain documents
    #  supports URL blacklist and whitelist filtering as well as a fixed URL list (items)
    if content_source['type'] != 'Website':
        raise (f"   Error loading documents. Cannot handle loading documents of type {content_source['type']}")

    site_urls = filtered_urls = None
    if 'location' in content_source:
        print(f"   Getting URLs from site map {content_source['location']}")
        site_urls = get_urls_from_sitemap(content_source['location'])

        # Apply filtering if defined.  First whitelist, then blacklist
        filtered_urls = site_urls
        if 'whitelist' in content_source:
            filtered_urls = filter_urls_with_whitelist(filtered_urls, content_source['whitelist'])

        if 'blacklist' in content_source:
            filtered_urls = filter_urls_with_blacklist(filtered_urls, content_source['blacklist'])

    else:
        site_urls = filtered_urls = content_source['items']

    print(
        f"   Found {len(site_urls)} URLs. Filted down to {len(filtered_urls)}. Lets go screen scraping (the slow part)")

    website_documents = clean_documents_newlines_spaces_and_tabs(scrape_website_pages(filtered_urls, headless))
    print(f"   ... and scraped pages resulting in {len(website_documents)} documents")

    if 'chunk_size' in content_source and content_source['chunk_size'] == 0:
        return website_documents
    else:
        webpage_chunks = chunk_up_documents(website_documents,
                                            content_source['chunk_size'] if 'chunk_size' in content_source else None)
        return webpage_chunks


class OpenSearchVectorDBLoader:
    __domain_name = None
    __region = None
    __index_name = None
    __embedding = None
    __data_sources = None

    def __init__(self, domain_name, index_name=None, embedding_model=None, data_sources=None, region=None):
        self.__domain_name = domain_name
        self.__region = region
        self.__index_name = index_name

        if embedding_model:
            self.__embedding_model = embedding_model
        else:
            self.__embedding_model = EmbeddingTypes.BEDROCK_DEFAULT

        self.__data_sources = data_sources

    def load_from_documents(self, documents, delete_index=False):
        load_dotenv(find_dotenv())
        embeddings = get_embeddings_from_model(self.__embedding_model)

        endpoint = get_opensearch_endpoint(self.__domain_name, self.__region)
        print(f"   Open search with endpoint={endpoint} and getting ready to load")
        print(f"   Index name = {self.__index_name} and loading {len(documents)} document chunks")

        credentials = boto3.Session().get_credentials()
        if self.__region:
            region = self.__region
        else:
            region = boto3.Session().region_name

        service = 'es'
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                           session_token=credentials.token)

        # Delete the index
        if delete_index:
            self.delete_index(self.__index_name)

        batch_size = 1000
        # Calculate the total number of batches needed
        total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)
        print(f"   Now loading {len(documents)} document chunks in {total_batches} batches of {batch_size}")
        for batch_num in range(total_batches):
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            document_subset = documents[start_index:end_index]

            db = OpenSearchVectorSearch.from_documents(
                document_subset,
                embeddings,  # Use the full embeddings list for each batch
                opensearch_url=f'https://{endpoint}',
                http_auth=awsauth,
                index_name=self.__index_name,
                connection_class=RequestsHttpConnection,
                verify_certs=True,
                bulk_size=batch_size
            )

            print(f"     Loaded batch {batch_num + 1} of {total_batches}")

        print(f"   Documents loaded into the DB")

        return db

    def load(self, content_sources=None, delete_index=True):
        # Get documents from each source and loads it into a vector DB
        #  Allow overriding the source and destination (index)
        #  BY default this will assume the index should be deleted and re-created.  This can be overridden
        #     by setting delete_index to False allowing multiple loads into the same index

        content_to_load = None
        if content_sources:
            content_to_load = content_sources
        else:
            if self.__data_sources:
                content_to_load = self.__data_sources
            else:
                raise (f"   Error loading documents. No content sources defined in constructor or method call")

        content_docs = None
        # delete_index = True  # in first item in loop, we clean, then we retain the index
        print(
            f"Going to load {len(self.__data_sources)} data sources into {self.__domain_name} domain and {self.__index_name} index")
        for content_source in content_to_load:
            print(f"Processing content for {content_source['name']} ")
            if content_source['type'] == 'Website':
                content_docs = get_documents_from_website(content_source)
            elif content_source['type'] in ['Microsoft Word', 'PDF']:
                content_docs = get_documents_from_folder(content_source)
            elif content_source['type'] == 'KustomerAPI':
                content_docs = get_all_shortcuts_as_documents(content_source['location'])
            else:
                raise (f"   Error loading documents. Cannot handle loading documents of type {content_source['type']}")

            print(f"Loading {len(content_docs)} content chunks from {content_source['name']} into DB")
            self.load_from_documents(content_docs, delete_index=delete_index)
            delete_index = False  # Now keep existing index data for each subsequent document set

    def delete_index(self, index_name):
        # This will delete the index if is exists.  If not, it will ignore the error
        #  requiring explicit definition of index, and not just class instance version - to avoid accidents
        credentials = boto3.Session().get_credentials()
        region = boto3.Session().region_name  # TODO: Handle this better - with region - leave as override or get from system
        service = 'es'
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                           session_token=credentials.token)

        endpoint = get_opensearch_endpoint(self.__domain_name, self.__region)

        # Delete the index
        print(f"   Deleting index={index_name}")
        delete_response = requests.request("DELETE", f'https://{endpoint}/{index_name}', auth=awsauth)
        if delete_response.status_code != 200:
            print(
                f"   Error deleting index={index_name}.  Likely not found and we can ignore. Status code={delete_response.status_code} Reason={delete_response.reason}")
        else:
            print(f"   Index deleted.")

    def get_index_name(self):
        return self.__index_name


class OpenSearchVectorDBQuery:
    __domain_name = None
    __region = None
    __index_name = None
    __embedding = None

    def __init__(self, domain_name, index_name, embedding_model=None, region=None):
        self.__domain_name = domain_name
        self.__region = region
        self.__index_name = index_name
        self.__embedding_model = embedding_model
        self.__client = None

    def get_client(self):
        if self.__client:
            return self.__client

        load_dotenv(find_dotenv())
        embeddings = get_embeddings_from_model(self.__embedding_model)
        endpoint = get_opensearch_endpoint(domain_name=self.__domain_name, region=self.__region)

        credentials = boto3.Session().get_credentials()
        region = boto3.Session().region_name
        service = 'es'
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                           session_token=credentials.token)

        self.__client = OpenSearchVectorSearch(
            embedding_function=embeddings,
            index_name=self.__index_name,
            opensearch_url=f'https://{endpoint}',
            http_auth=awsauth,
            use_ssl=True,
            connection_class=RequestsHttpConnection,
            verify_certs=True,
            timeout=30
        )

        return self.__client

    def query(self, query, result_set=5):
        vdb = self.get_client()
        query_results = vdb.similarity_search(query, k=result_set)

        return query_results

    def get_index_name(self):
        return self.__client.get_index_name()
