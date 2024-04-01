import unittest
import os
from dotenv import find_dotenv, load_dotenv
from aws_rag_bot.aws_opensearch_vector_database import (
    get_embeddings_from_model,
    EmbeddingTypes,
    OpenSearchVectorDBQuery
)
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from pprint import pprint

from aws_rag_bot.kustomer import get_all_shortcuts_as_documents

load_dotenv(find_dotenv())


# domain_name = os.getenv("OPENSEARCH_DOMAIN")

class TestOpenSearchVectorDBLoader(unittest.TestCase):

    def test_get_embeddings_from_model_default(self):
        embedding = get_embeddings_from_model()
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), BedrockEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_bedrock_cohere(self):
        embedding = get_embeddings_from_model(EmbeddingTypes.BEDROCK_COHERE)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), BedrockEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_openai(self):
        embedding = get_embeddings_from_model(EmbeddingTypes.OPENAI_GPT_DEFAULT)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), OpenAIEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_openai_custom(self):
        embedding_model = {"name": "openai-text-embed3-large", "provider": "openai", "model": "text-embedding-3-large"}
        embedding = get_embeddings_from_model(embedding_model)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), OpenAIEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_huggingface_default(self):
        embedding = get_embeddings_from_model(EmbeddingTypes.HUGGING_FACE_DEFAULT)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), HuggingFaceEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_huggingface_custom(self):
        embedding_model = {"name": "hugging-face-bge-large", "provider": "hugging-face",
                           "model": "BAAI/bge-large-en-v1.5"}
        embedding = get_embeddings_from_model(embedding_model)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), HuggingFaceEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_get_embeddings_from_model_cohere(self):
        embedding = get_embeddings_from_model(EmbeddingTypes.COHERE_DEFAULT)
        self.assertTrue(embedding)
        self.assertEqual(type(embedding), CohereEmbeddings)
        embedding_vector = embedding.embed_query("What is the meaning of life?")
        print(f"embedding_length: {len(embedding_vector)}")
        self.assertGreater(len(embedding_vector), 0)

    def test_query(self):
        load_dotenv(find_dotenv())
        endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        vdb = OpenSearchVectorDBQuery(os_endpoint=endpoint, service='aoss', index_name='index-artemis-mission')
        documents = vdb.query("Where is the Artemis program going")
        for doc in documents:
            print("----------------------- metadata -------------------------")
            pprint(doc.metadata)
            print("--------------------- page content -----------------------")
            pprint(doc.page_content)

        self.assertGreater(len(documents), 0)

    def test_kustomer_get_documents(self):
        content_source = {"name": "Kustomer Shortcuts", "type": "KustomerAPI", "location": "v1/shortcuts", "chunk_size": 1024}
        documents = get_all_shortcuts_as_documents(content_source)
        print(f"Found {len(documents)} documents")
        self.assertGreater(len(documents), 0)

        max_doc_size = 0
        for doc in documents:
            if len(doc.page_content) > max_doc_size:
                max_doc_size = len(doc.page_content)

        print(f"Max document size: {max_doc_size}")



# get_urls_from_sitemap
# Test against a known sitemap and make sure count is greater than a value

# scrape_website_pages
# Test a small known website
# use blacklist
# use whitelist
# use chunk size

# get_documents_from_folder
# filter_urls_with_blacklist
# filter_urls_with_whitelist
# chunk_up_documents
# get_documents_from_website

# OpenSearchVectorDBLoader
# load_from_documents
# load
# delete_index
# get_index_name

# Test load and search for a known document

if __name__ == '__main__':
    unittest.main()
