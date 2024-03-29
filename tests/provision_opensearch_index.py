from aws_rag_bot.aws_opensearch_vector_database import OpenSearchVectorDBLoader, EmbeddingTypes
from dotenv import find_dotenv, load_dotenv
import os

artemis_url_list = [
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://www.nasa.gov/humans-in-space/artemis/",
    "https://www.nasa.gov/mission/artemis-i/",
    "https://www.nasa.gov/mission/artemis-ii/",
    "https://www.nasa.gov/directorates/esdmd/common-exploration-systems-development-division/space-launch-system/rocket-propellant-tanks-for-nasas-artemis-iii-mission-take-shape/",
    "https://www.nasa.gov/centers-and-facilities/hq/splashdown-nasas-orion-returns-to-earth-after-historic-moon-mission/",
    "https://www.nasa.gov/missions/artemis/nasas-first-flight-with-crew-important-step-on-long-term-return-to-the-moon-missions-to-mars/",
    "https://www.nasa.gov/missions/artemis/artemis-iii/",
    "https://www.rmg.co.uk/stories/topics/nasa-moon-mission-artemis-program-launch-date",
    "https://www.asc-csa.gc.ca/eng/astronomy/moon-exploration/artemis-missions.asp",
    "https://www.space.com/nasa-artemis-2-moon-mission-delay-september-2025"
]

content_source = {"name": "Reports", "type": "Website", "items": artemis_url_list}

content_sources = [content_source]  # This is an array, because it could be multiple sources

# Make sure you have an opensearch domain setup at AWS, the domain name is in the .env file and
# you have the proper credentials in the .aws/credentials file to access it
load_dotenv(find_dotenv())
#my_open_search_domain_name = os.getenv("OPENSEARCH_DOMAIN")
my_open_search_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
my_index_name = 'index-artemis-mission'

# vectordb_loader = OpenSearchVectorDBLoader(domain_name=my_open_search_domain_name,
#                                            index_name=my_index_name,
#                                            embedding_model=EmbeddingTypes.BEDROCK_DEFAULT,
#                                            data_sources=content_sources)

vectordb_loader = OpenSearchVectorDBLoader(os_endpoint=my_open_search_endpoint,
                                           service='aoss',  #  aoss for serverless, es for instance
                                           index_name=my_index_name,
                                           embedding_model=EmbeddingTypes.BEDROCK_DEFAULT,
                                           data_sources=content_sources)

vectordb_loader.load()
