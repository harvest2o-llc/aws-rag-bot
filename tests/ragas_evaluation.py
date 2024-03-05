import os
from dotenv import find_dotenv, load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

from aws_rag_bot.rag_chatbot import RagChatbot, LlmModelTypes
from aws_rag_bot.prompt_library import NasaSpokespersonPrompts
from aws_rag_bot.aws_opensearch_vector_database import EmbeddingTypes, OpenSearchVectorDBQuery

# The basic idea here:
# 1. Create a set of test questions and answers (ground truth's) to test.  This can be done synthetically, but I chose manual
# 2. Run your questions through the RAG bot to fill out bot answers and content returned during retrieval
# 3. Then run the data set (questions, answers, content and ground truths) through the RAGAS evaluation tool to get scores

# This is just a basic example.  From here you would build a full set of test questions
#    then run them through variants of LLM model, model parameters, embeddings, prompt models, etc.
#    to compare performance

questions_and_ground_truths = [
    {"question": "Under what president was the Artimis program established.",
     "ground_truth": "Vice President Pence announced the Artemis program on March 26, 2019 with Trump as president."},

    {"question": "When was the first Atemis mission launched?", "ground_truth": "November 16, 2021"},

    {"question": "What spacecraft was used for the Artimis 1 mission?",
     "ground_truth": "Orion"},
    {"question": "Which mission will have the first astronauts aboard?",
     "ground_truth": "Artemis III will have four astronauts aboard."},

    {"question": "Why are private companies involved in the Artemis program?",
     "ground_truth": "Private companies are involved to help promote innovation, lower costs and lower the risk of program success."}
    ]

# Build the input data set for evaluation
questions = []
ground_truths = []
contexts = []
answers = []

# Build it in the shape that RAGAS expects
for q in questions_and_ground_truths:
    questions.append(q["question"])
    ground_truths.append(q["ground_truth"])

load_dotenv(find_dotenv())
domain_name = os.getenv('OPENSEARCH_DOMAIN')
llm_model_type = LlmModelTypes.BEDROCK_TITAN_EXPRESS
prompt_model = NasaSpokespersonPrompts


# Get an LLM model from our chatbot
chatbot = RagChatbot(domain_name,
                     llm_model_type,
                     prompt_model,
                     EmbeddingTypes.BEDROCK_DEFAULT
                     )
llm = chatbot.get_llm_model()

# Get a retriever to use for the process
vector_db = OpenSearchVectorDBQuery(domain_name=domain_name, index_name=prompt_model.llm_index)
vector_db_client = vector_db.get_client()
retriever = vector_db_client.as_retriever()

# Get answers and content from our RAbot
for query in questions:
    answer = chatbot.ask_question(query)
    answers.append(answer.content.strip())  # Strip off any leading or trailing white space
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict for proper shaping to evaluate
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset_dict = Dataset.from_dict(data_dict)

# Now we have all the data we need to run the evaluation
    # https://docs.ragas.io/en/stable/references/evaluation.html
results_from_eval = evaluate(
    dataset=dataset_dict,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    is_async=False,
    raise_exceptions=False  # If exceptions found, which is likely, don't raise them (they will get logged)
)

print(results_from_eval)