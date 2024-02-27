import requests
from langchain_core.documents import Document
import boto3
import os
from dotenv import load_dotenv, find_dotenv


def get_all_shortcuts_as_documents(kustomer_function):
    # Get additional configuration
    load_dotenv(find_dotenv())
    kustomer_api_key = os.getenv('KUSTOMER_API_KEY')

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

    return documents
