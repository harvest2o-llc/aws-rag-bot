from dotenv import find_dotenv, load_dotenv
import os

import unittest
from colorama import Fore, Back, Style
from langchain_core.messages import HumanMessage

from aws_rag_bot.prompt_library import *
from aws_rag_bot.rag_chatbot import RagChatbot, LlmModelTypes


load_dotenv(find_dotenv())
endpoint = os.getenv("OPENSEARCH_ENDPOINT")
question = "what is the Answer to the Ultimate Question of Life, the Universe, and Everything"
question2 = "what is a typical color of the sky?"

context_question = "How many astronauts are going to be the first to land on the moon?"

conv_question1 = "How many astronauts are going to be the first to land on the moon?"
conv_question2 = "What are their names?"

verbose = False

class TestRagbot(unittest.TestCase):
    def test_bedrock_titan(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_TITAN_EXPRESS)
        llm = chatbot.get_llm_model()
        print(Fore.BLUE + f"question: {question}" + Style.RESET_ALL)
        print()
        meaning_of_life = llm.invoke(question + "\n")
        print(Fore.GREEN + "Answer:\n" + meaning_of_life + Style.RESET_ALL)
        print(f"Response type: {type(meaning_of_life)}")
        self.assertIn("42", meaning_of_life)

    def test_bedrock_llama2(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_LLAMA2)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_bedrock_jurassic(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_JURRASIC2_ULTRA)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_bedrock_claude_instant(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_INSTANT)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_bedrock_claude21(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)


    def test_bedrock_claude3_sonnet(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE3_SONNET)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life.content)
        self.assertIn("42", meaning_of_life.content)


    def test_bedrock_claude21_optimized_prompt(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()

        # Claud 2.1 has some specialized prompt rules that help.
        # https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources
        claude21_question = """
        
        
        Human: What is the Answer to the Ultimate Question of Life, the Universe, and Everything
        
        Assistant:
        """

        meaning_of_life = llm.invoke(claude21_question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_openai_gpt4(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT4)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life.content)
        self.assertIn("42", meaning_of_life.content)

    def test_openai_gpt3(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT35)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life.content)
        self.assertIn("42", meaning_of_life.content)

    def test_google_gemini_pro(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.GOOGLE_GEMINI_PRO)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life.content)
        self.assertIn("42", meaning_of_life.content)

    # Test chatob with basic questions
    def test_chatbot_with_titan(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_TITAN_EXPRESS, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)

        self.assertIsNotNone(response)

    def test_chatbot_with_llama2(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_LLAMA2, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_jurassic(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_JURRASIC2_ULTRA, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_claude_instant(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_INSTANT, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_claude21(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_21, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_claude3_sonnet(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE3_SONNET, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_openai_gpt4(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT4, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_openai_gpt3(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT35, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_with_google_gemini_pro(self):
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.GOOGLE_GEMINI_PRO, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {context_question}" + Style.RESET_ALL)
        print()
        response = chatbot.ask_question(context_question, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response + Style.RESET_ALL)
        self.assertIsNotNone(response)

    def test_chatbot_conversation_with_titan(self):
        # NOTE: This test does not work on Titan, but does on all others. Still investigating.
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_TITAN_EXPRESS, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_llama2(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_LLAMA2, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_jurassic(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_JURRASIC2_ULTRA, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_claude_instant(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_INSTANT, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)


    def test_chatbot_conversation_with_claude21(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE_21, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_claude3_sonnet(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.BEDROCK_CLAUDE3_SONNET, prompt_model=ClaudNasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)


    def test_chatbot_conversation_with_openai_gpt4(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT4, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_openai_gpt3(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.OPENAI_GPT35, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)

    def test_chatbot_conversation_with_google_gemini_pro(self):
        chat_history = []
        chatbot = RagChatbot(endpoint, model_key=LlmModelTypes.GOOGLE_GEMINI_PRO, prompt_model=NasaSpokespersonPrompts)
        print(Fore.BLUE + f"Question: {conv_question1}" + Style.RESET_ALL)
        print()
        response1 = chatbot.ask_question(conv_question1, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response1 + Style.RESET_ALL)
        self.assertIsNotNone(response1)

        print("----------------------------------------------------------------------")
        chat_history.extend([{"question": conv_question1, "response": response1}])
        print(Fore.BLUE + f"Question: {conv_question2}" + Style.RESET_ALL)
        print()
        response2 = chatbot.ask_question(conv_question2, chat_history, verbose=verbose)
        print(Fore.GREEN + "Answer:\n" + response2 + Style.RESET_ALL)
        self.assertIsNotNone(response2)