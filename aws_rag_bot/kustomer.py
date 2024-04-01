import requests
from langchain_core.documents import Document
import boto3
import os
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_up_shortcuts(documents, chunk_size):
    if chunk_size is None:
        chunk_size = 512

    if chunk_size == 1:
        return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=round(chunk_size * 0.15),
        # I chose 15% overlap because that is what I use in woodworking scrap estimate  :-)
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"   Chunked {len(documents)} documents into {len(docs_chunks)} chunks using size={chunk_size}")

    return docs_chunks


def get_all_shortcuts_as_documents(content_source):
    # Get additional configuration
    load_dotenv(find_dotenv())
    kustomer_api_key = os.getenv('KUSTOMER_API_KEY')
    kustomer_function = content_source['location']

    base_url = "https://api.kustomerapp.com"
    initial_url = f"{base_url}/{kustomer_function}"
    headers = {
        "Authorization": f"Bearer {kustomer_api_key}",
        "Content-Type": "application/json"
    }

    print(f"Fetching shortcuts from {initial_url}")
    documents = []
    url = initial_url
    while url:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            for item in data.get('data', []):
                # Check if appDisabled or deleted are False
                if item['attributes'].get('appDisabled', True) or item['attributes'].get('deleted', True):
                    continue  # Skip this item

                name = item['attributes'].get('name', '')
                text = item['attributes']['draft'].get('text', '')
                page_content = f"{name}: {text}"
                metadata = {
                    "source": "Kustomer Shortcut",
                    "title": name,
                    "created_at": item['attributes'].get('updatedAt', '')
                }
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

            # Check if there's a next page
            next_link = data.get('links', {}).get('next')
            url = f"{base_url}{next_link}" if next_link else None
        else:
            raise Exception(f"Failed to fetch shortcuts. Status code: {response.status_code}")

    if 'chunk_size' in content_source and content_source['chunk_size'] == 0:
        return documents
    else:
        doc_chunks = chunk_up_shortcuts(documents,
                                        content_source['chunk_size'] if 'chunk_size' in content_source else None)
        return doc_chunks
