import unittest

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig

from aws_rag_bot.prompt_library import NasaSpokespersonPrompts  # , ClaudNasaSpokespersonPrompts
from aws_rag_bot.rag_chatbot import RagChatbot, LlmModelTypes
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())
domain_name = os.getenv("OPENSEARCH_DOMAIN")
question = "what is the Answer to the Ultimate Question of Life, the Universe, and Everything"
context_question = "How many astronauts are going to the moon?"

conv_question1 = "How many astronauts are going to the moon?"
conv_question2 = "What are their names?"

# Template for Claude

task_content = """
You are the spokesperson for NASA named Sally speaking to a  \
    group of journalists about the Artemis space program. \
    \
    keep the answer concise.\
"""

tone_and_context = """Keep the tone friendly, positive, informative and engaging. """

rules_and_guidelines = """
If the question is not related to NASA or the Artemis program, 
just say I am here to talk about NASA and our mission to the moon.
Just provide the answer and don't follow the answer with a question to help more
"""

examples = """"""

# History goes here
conversation_history_summary = "Current conversation:\n<conversational_history>{chat_history}</conversational_history>"
conversation_history_summary = ""

# retrieval document chunks go here
supporting_documents = "Answer the question based only on the following context: \n{context}"

# Ask for proof of reference to help
proof = """Please provide the most relevant sentences that support your answer inside <proof> tags."""

# Then the question :)
human_question = "{question}"

# Bring it all together
# total_prompt = f"\n\nHuman: {task_content}\n{tone_and_context}\n{rules_and_guidelines}\n{examples}\n{conversation_history_summary}\n{supporting_documents}\n{proof}\n{human_question}\nAssistant:"
total_prompt = f"{task_content}\n{tone_and_context}\n{rules_and_guidelines}\n{examples}\n{conversation_history_summary}\n{supporting_documents}\n{proof}\n{human_question}\nAssistant:"


class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, input, prompts, **kwargs):
        print("--------------------- Inspect Prompts --------------------------------------")
        print(prompts)
        print("---------------------------------------------------------------------------")

    def on_llm_end(self, output, **kwargs):
        results = output.flatten()
        print("--------------------- Inspect Results --------------------------------------")
        print(results)
        print("---------------------------------------------------------------------------")

class TestClaude21(unittest.TestCase):

    def test_bedrock_claude_instant(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_INSTANT)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_bedrock_claude21(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()
        meaning_of_life = llm.invoke(question)
        print(meaning_of_life)
        self.assertIn("42", meaning_of_life)

    def test_bedrock_claude21_optimized_prompt(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()

        # Claud 2.1 has some specialized prompt rules that help.
        # https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources
        claude21_question = total_prompt
        print(claude21_question)
        print()
        print("--------------------------------------------------------------------------------------------------")

        answer = llm.invoke(claude21_question)
        print(f"Type of response from Claude: {type(answer)}")
        print(answer)

    def test_bedrock_claude21_query_vdb(self):
        # Create a callback handler for a bit more debugging

        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()
        # Get a retriever
        vdb = chatbot.get_vector_db()
        retriever = vdb.as_retriever()

        # Register my callback handler
        my_callback_handler = MyCallback()
        chain_config = RunnableConfig(callbacks=[my_callback_handler])

        template = total_prompt
        prompt = ChatPromptTemplate.from_template(template)

        # This takes the context from the retrirver, based on the question, then
        #   passes the resulting context and question to the prompt template
        #   then executes the query and returns.
        my_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        my_chain = my_chain.with_config(chain_config)

        print()
        the_question = "How many astronauts are going to be on the first crewed moon landing?"
        print(f"Question: {the_question}")
        print("--------------------------------------------------------------------------------------------------")
        answer = my_chain.invoke(the_question)
        print("-------------------------------------- the answer ------------------------------------------------")
        print(answer)

    def test_bedrock_claude21_query_vdb_with_hist(self):
        # Create a callback handler for a bit more debugging

        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()
        # Get a retriever
        vdb = chatbot.get_vector_db()
        retriever = vdb.as_retriever()

        # Register my callback handler
        my_callback_handler = MyCallback()
        chain_config = RunnableConfig(callbacks=[my_callback_handler])

        template = total_prompt
        prompt = ChatPromptTemplate.from_template(template)

        # This takes the context from the retriever, based on the question, then
        #   passes the resulting context and question to the prompt template
        #   then executes the query and returns.
        history = []

        my_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        my_chain = my_chain.with_config(chain_config)

        print()
        the_question = "How many astronauts are going to be on the first crewed moon landing?"
        print(f"Question: {the_question}")
        print("--------------------------------------------------------------------------------------------------")
        answer = my_chain.invoke(the_question)
        print("-------------------------------------- the answer ------------------------------------------------")
        print(answer)

        print()
        print("==========================  Now Go at it with a follow-up question ===============================")
        # first we want to summarize the history, using a different chain
        conversation_summary_prompt = """Given a chat history which you can find in the <conversational_history> tags below
            and the latest my latest question which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.
            
            <conversational_history>{chat_history}</conversational_history>
            
            and here is my latest question
            
            {question}
            """

        chat_history = f"Human: {the_question}\nAssistant: {answer}"
        next_question = "What are their names?"
        summary_prompt = ChatPromptTemplate.from_template(conversation_summary_prompt)
        assembled_prompt = summary_prompt.format(chat_history=chat_history, question=next_question)

        history_summary_chain = (
                assembled_prompt
                | llm
                | StrOutputParser()
        )

        history_summary_chain = history_summary_chain.with_config(chain_config)

        print(f"Next Question: {next_question}")
        print("--------------------------------------------------------------------------------------------------")
        next_answer = history_summary_chain.invoke(next_question)
        print("------------------------------- the second answer ------------------------------------------------")
        print(next_answer)

    def test_print_qa_prompt(self):
        my_qa_system_prompt = ClaudNasaSpokespersonPrompts.qa_prompt
        my_message = my_qa_system_prompt.format(question='What is your name?', context='foo', chat_history=['bar'])
        print(my_message)

    def test_print_contextualize_q_prompt(self):
        my_qa_system_prompt = ClaudNasaSpokespersonPrompts.contextualize_q_system_prompt
        my_message = my_qa_system_prompt.format(question='What is your name?', context='foo', chat_history=['bar'])
        print(my_message)

    def test_bedrock_claude21_adhoc(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        llm = chatbot.get_llm_model()

        # Claud 2.1 has some specialized prompt rules that help.
        # https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources
        claude21_question = """


    Human: Can you tell me your favorite color?
    
    Assistant: Well.... I don't exactly have a favorite color because I can't see ;) But there are some colors I like.
    
    Human: OK, what's a color you like?
    
    Assistant:
                """

        meaning_of_life = llm.invoke(claude21_question)
        print(f"Type of response from Claude: {type(meaning_of_life)}")
        print(meaning_of_life)

    def test_chatbot_with_claude_instant(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_INSTANT)
        response = chatbot.ask_question(context_question)
        self.assertIsNotNone(response.content)

    def test_chatbot_with_claude21(self):
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21)
        response = chatbot.ask_question(context_question)
        self.assertIsNotNone(response.content)

    def test_chatbot_conversation_with_claude_instant(self):
        chat_history = []
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_INSTANT, prompt_model=NasaSpokespersonPrompts)
        response1 = chatbot.ask_question(conv_question1, chat_history)
        print(response1.content)
        self.assertIsNotNone(response1.content)
        chat_history.extend([HumanMessage(content=conv_question1), response1])
        response2 = chatbot.ask_question(conv_question2, chat_history)
        print(response2.content)
        self.assertIsNotNone(response2.content)

    def test_chatbot_conversation_with_claude21(self):
        chat_history = []
        chatbot = RagChatbot(domain_name, LlmModelTypes.BEDROCK_CLAUDE_21, prompt_model=NasaSpokespersonPrompts)
        response1 = chatbot.ask_question(conv_question1, chat_history)
        print(response1.content)
        self.assertIsNotNone(response1.content)
        chat_history.extend([HumanMessage(content=conv_question1), response1])
        response2 = chatbot.ask_question(conv_question2, chat_history)
        print(response2.content)
        self.assertIsNotNone(response2.content)
