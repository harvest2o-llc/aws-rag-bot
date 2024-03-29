import argparse
from colorama import Fore, Back, Style
from aws_rag_bot.prompt_library import *
from aws_rag_bot.rag_chatbot import RagChatbot, LlmModelTypes
from dotenv import find_dotenv, load_dotenv
import os

# Load the environment variables as an option
load_dotenv(find_dotenv())
default_endpoint = os.getenv('OPENSEARCH_ENDPOINT', default=None)

parser = argparse.ArgumentParser(description='Chatbot CLI Client')
parser.add_argument('-e', '--endpoint', type=str, help='The endpoint for the vector DB', default=default_endpoint)
args = parser.parse_args()

endpoint = args.endpoint
if not endpoint:
    print("Please provide an OpenSearch endpoint for the chatbot in command line argument or in the .env file.")
    exit(1)
else:
    print(f"Using OpenSearch domain name: {endpoint}")

chatbot = RagChatbot(endpoint, service_name='aoss',  # hardedcoded here, so change if using instance to "es"
                     model_key=LlmModelTypes.BEDROCK_CLAUDE_21,
                     prompt_model=ClaudNasaSpokespersonPrompts)
chat_history = []

while True:
    human_input = input(Fore.RED + "Hello, how can I help you? (x or exit to end this conversation)\n" + Style.RESET_ALL)
    if human_input.lower() in ["exit", "quit", "bye", "goodbye", "x", "q"]:
        print()
        print("I'm out of here!")
        break
    response = chatbot.ask_question(human_input, chat_history)

    # Keep track of the conversation history for future questions
    chat_history.extend([{"question": human_input, "response": response}])

    print(Fore.BLUE + response + Style.RESET_ALL)
    print()

#Now print out the final summary of cost and tokens
conversational_summary = chatbot.get_model_run_summary()
print("========================= Summary of the conversation =========================")
print(f"Total Input tokens:  {conversational_summary['total_input_tokens']}")
print(f"      Output tokens: {conversational_summary['total_output_tokens']}")
print(f"      Cost:          ${conversational_summary['total_cost']:.4f}")
print(f"LLM Model used:  {conversational_summary['model_name']['name']}")
print(f"Vector DB index: {conversational_summary['vector_db_index']}")


