## AWS OpenSearch Vector Database and LangChain
The aws_openssaerch_vector.py includes classes and utility functions that can make it easy to connect to
and load multiple content sources into an OpenSearch vector database.

Part of the goal in this was to create a way to define multiple sources of data to load
with settings specific to each source to fine tune the content to best optimize the RAG app.

This library is meant more to provide a complete working set of examples and less to be an out-of-the-box library to use as-is.

### Features
- Array of data sources to drive the loading process
   - Sources can be entire websites (driven by providing a site map) or a list of URLs
     - Whitelist and blacklist specification can be used to get finer control 
   - Files in local directory (soon to support S3 and Google Drive)
   - Control of document chunk sizes by source
- This supports multiple embedding types (with more to come or you modify for your own needs)
  - Bedrock (Titan)
  - OpenAI (default)
  - HuggingFace (default - local)
- Uses the LangChain SeleniumURLLoader to scrape websites and provide human-readable content (instead of just the HTML)

### How to use
The library can be used at the highest level of functionality - which is to instantiate the class and then call the load method
with a list of sources.   OR you can use the lower level functions to play work with the loaders and splitters directly.

You can also get the vector DB client and use the classes on that directly.

#### Prerequisites
- Python 3.11 (may work on older, but this was the target version used)
- AWS Account  - account information expected in the .aws/credentials file
- OpenSearch Domain created and accessible
- Access to an embedding model (Bedrock Titan, OpenAI, HuggingFace) - HuggingFace works locally, the others are cloud-based and require an account and keys

#### Sample Code
You can find some sample code in the [test_aws_opensearch_vector.py](test_aws_opensearch_vector.py) file.  
Here is a simple example to load a list of URLs into a vector database:
```python
from aws_opensearch_vector_database import OpenSearchVectorDBLoader

content_source = [{"name": "Reports", "type": "Website", "items": url_list}]

vectordb_loader = OpenSearchVectorDBLoader(domain_name='my-os-domain-name',  
                                     index_name='my-index-name',
                                     data_sources=content_sources)

vectordb_loader.load()
```
Then you can query it like this: 
```python
from aws_opensearch_vector_database import OpenSearchVectorDBQuery
vectordb = OpenSearchVectorDBQuery(domain_name='my-os-domain-name', index_name='my-index-name')
results = vectordb.query("semantic query text here")

```
#### Defining Data Sources
```json
{
  "name": "Name of source", 
  "type": "connection_type", 
  "location": "path_or_url", 
  "items": [],
  "blacklist": [],
  "whitelist": [],
  "chunk_size": 0
}
```
Where
- name - a name is whatever you want to call the source.  Only meant for logging output
- type - the type of source.  Currently supports "Website", "Microsoft Work" and "PDF".  I have a custom API call but that is bespoke to my environment and is likely not useful
- location - If using website, this will be a URL to the sitemap.  If it is a local directly, it will be a relative path to the directory
- items - For websites, you can exclude the location (sitemap) and provide a fixed list of URLs' to load.
- blacklist - If using type of Website, this can be an array of wildcard based definitions of what to exclude in the website of URLs (soon to support directory loaders too)
- whitelist - If using type of Website, this can be an array of wildcard based definition of what to include in the website of URLs (soon to support directory loaders too)
- chunk_size - This is passed through to the LangChain RecursiveCharacterTextSplitter for loading the vector DB.  Read the Langchain docs for more information on this.


### What's next (Roadmap):
Future
1. Finish whitelist and blacklist for directory loads
2. Add OpenSearch serverless to the class as an option
2. Add an S3 directory loader
3. Add a Google Drive loader
5. Create a simple test file to run through tests of mine... validating each major function


https://blog.langchain.dev/improving-document-retrieval-with-contextual-compression/

