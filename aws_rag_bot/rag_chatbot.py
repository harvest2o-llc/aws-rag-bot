from datetime import datetime
import boto3
from dotenv import find_dotenv, load_dotenv

from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler

# Code in this project
from aws_rag_bot.aws_opensearch_vector_database import (
    EmbeddingTypes,
    OpenSearchVectorDBQuery
)
from aws_rag_bot.prompt_library import *


class LlmModelTypes:
    BEDROCK_LLAMA2 = "bedrock_llama2"
    BEDROCK_JURRASIC2_ULTRA = "bedrock_jurrasic2_ultra"
    BEDROCK_TITAN_EXPRESS = "bedrock_titan_express"
    BEDROCK_CLAUDE_INSTANT = "bedrock_claude_instant"
    BEDROCK_CLAUDE_21 = "bedrock_claude21"
    BEDROCK_CLAUDE3_SONNET = "bedrock_claude3_sonnet"
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    GOOGLE_GEMINI_PRO = "google_gemini_pro"


def structure_message_history(chat_history):
    # format of chat history is: {"question": "****", "response": "****"]
    if chat_history is None or len(chat_history) == 0:
        return []
    else:
        history_string = []
        for chat_tuple in chat_history:
            history_string.append(HumanMessage(content=chat_tuple["question"]))
            history_string.append(AIMessage(content=chat_tuple["response"]))
        return history_string


class RagChatbot:
    __llm_model = None
    __conversation_chain = None
    __vector_db = None
    __vector_db_embedding = None

    __chat_prompt = None
    __log_verbose = False
    __total_cost = 0
    __total_input_tokens = 0
    __total_output_tokens = 0
    __total_run_duration = 0
    __last_run_prompt = None
    __last_run_input_tokens = 0
    __last_run_output_tokens = 0
    __last_run_cost = 0
    __last_run_duration = 0

    __bedrock_model_def_llama2 = {
        "key": "bedrock_llama2",
        "name": "Bedrock Llama 2",
        "id": "meta.llama2-70b-chat-v1",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "max_gen_len": 500,
            "temperature": 0.2,
            "top_p": 0.2,
        },
        "model_cost": {
            "input_token_cost": 0.00195 / 1000,
            "output_token_cost": 0.00256 / 1000
        }
    }

    __bedrock_model_def_jurrasic2_ultra = {
        "key": "bedrock_jurrasic2_ultra",
        "name": "Bedrock Jurrasic 2 Ultra",
        "id": "ai21.j2-ultra-v1",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "maxTokens": 500,
            "temperature": 0.5,
            "topP": 0.5,
        },
        "model_cost": {
            "input_token_cost": 0.0188 / 1000,
            "output_token_cost": 0.0188 / 1000
        }
    }

    __bedrock_model_def_titan_express = {
        "key": "bedrock_titan_express",
        "name": "Bedrock Titan Express",
        "id": "amazon.titan-text-express-v1",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "maxTokenCount": 500,
            "temperature": 0.1,
            "topP": 0.5,
        },
        "model_cost": {
            "input_token_cost": 0.0008 / 1000,
            "output_token_cost": 0.0016 / 1000
        }
    }

    __bedrock_model_def_claude_instant = {
        "key": "bedrock_claude_instant",
        "name": "Bedrock Claude Instant",
        "id": "anthropic.claude-instant-v1",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "max_tokens_to_sample": 500,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 250,
        },
        "model_cost": {
            "input_token_cost": 0.00080 / 1000,
            "output_token_cost": 0.00240 / 1000
        }
    }

    __bedrock_model_def_claude21 = {
        "key": "bedrock_claude21",
        "name": "Bedrock Claude v2.1",
        "id": "anthropic.claude-v2:1",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "max_tokens_to_sample": 500,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 250,
        },
        "model_cost": {
            "input_token_cost": 0.00080 / 1000,
            "output_token_cost": 0.00240 / 1000
        }
    }


    # https://medium.com/@dminhk/building-with-anthropics-claude-3-on-amazon-bedrock-and-langchain-️-2b842f9c0ca8
    __bedrock_model_def_claude3_sonnet = {
        "key": "bedrock_claude3_sonnet",
        "name": "Bedrock Claude v3 Sonnet",
        "id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "client_name": "bedrock-runtime",
        "region_name": "us-east-1",
        "kwargs": {
            "max_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 250,
            "stop_sequences": ["\n\nHuman"],
        },
        "model_cost": {
            "input_token_cost": 0.00300 / 1000,
            "output_token_cost": 0.01500 / 1000
        }
    }


    __open_ai_model_def_gpt4 = {
        "key": "openai_gpt4",
        "name": "OpenAI GPT-4",
        "id": "gpt-4",
        "client_name": "openai",
        "kwargs": {
            "maxTokens": 500,
            "temperature": 0.5,
        },
        "model_cost": {
            "input_token_cost": 0.03 / 1000,
            "output_token_cost": 0.06 / 1000
        }
    }

    __open_ai_model_def_gpt35 = {
        "key": "openai_gpt35",
        "name": "OpenAI GPT-3.5 Turbo",
        "id": "gpt-3.5-turbo",
        "client_name": "openai",
        "kwargs": {
            "maxTokens": 500,
            "temperature": 0.5,
        },
        "model_cost": {
            "input_token_cost": 0.0005 / 1000,
            "output_token_cost": 0.0015 / 1000
        }
    }

    __google_gemini_pro = {
        "key": "google_gemini_pro",
        "name": "Google Gemini Pro",
        "id": "gemini-pro",
        "client_name": "google",
        "kwargs": {
            "maxTokens": 500,
            "temperature": 0.5,
        },
        "model_cost": {  # uses cost/char... estimated at 4 chars per token
            "input_token_cost": 0.000125 * 4 / 1000,
            "output_token_cost": 0.000375 * 4 / 1000
        }
    }

    __model_options = [
        __bedrock_model_def_llama2,
        __bedrock_model_def_jurrasic2_ultra,
        __bedrock_model_def_titan_express,
        __bedrock_model_def_claude_instant,
        __bedrock_model_def_claude21,
        __bedrock_model_def_claude3_sonnet,
        __open_ai_model_def_gpt4,
        __open_ai_model_def_gpt35,
        __google_gemini_pro
    ]
    __current_model = None
    __prompt_model = None

    def token_count(self, string):
        return self.__llm_model.get_num_tokens(string)

    # Constructor
    def __init__(self, vector_db_domain
                 , model_key=LlmModelTypes.BEDROCK_TITAN_EXPRESS
                 , prompt_model=None
                 , embedding_model=EmbeddingTypes.BEDROCK_DEFAULT
                 , verbose=False):

        load_dotenv(find_dotenv())
        if prompt_model is None:
            self.__prompt_model = BasePromptModel()
        else:
            self.__prompt_model = prompt_model

        self.__log_verbose = verbose

        # Set the current model based on input
        for model_option in self.__model_options:
            if model_option['key'] == model_key:
                self.__current_model = model_option
                break

        # Set the index name based on the prompt model
        index_name = self.__prompt_model.llm_index

        # Initialize the LLM model, memory, and chain
        self.__llm_model = self.__get_llm_model()
        vector_db = OpenSearchVectorDBQuery(domain_name=vector_db_domain,
                                            index_name=index_name,
                                            embedding_model=embedding_model)
        self.__vector_db = vector_db.get_client()

    class RagCallback(BaseCallbackHandler):
        # https://how.wtf/how-to-count-amazon-bedrock-anthropic-tokens-with-langchain.html
        def __init__(self, llm, verbose=False):
            self.llm = llm
            self.prompt = None
            self.input_tokens = 0
            self.output_tokens = 0
            self.verbose = verbose

        def on_llm_start(self, input, prompts, **kwargs):
            self.prompt = prompts
            if self.verbose:
                print("--------------------- Callback On Start LLM Prompts --------------------------------------")
                print(prompts)
                print("------------------------------------------------------------------------------------------")
            for p in prompts:
                self.input_tokens = self.llm.get_num_tokens(p)

        def on_llm_end(self, output, **kwargs):
            results = output.flatten()
            if self.verbose:
                print("--------------------- Callback On End LLM Results --------------------------------------")
                print(results)
                print("----------------------------------------------------------------------------------------")
            for r in results:
                self.output_tokens = self.llm.get_num_tokens(r.generations[0][0].text)

        def cost(self, model_info):
            if 'model_cost' in model_info:
                return self.input_tokens * model_info['model_cost']['input_token_cost'] + self.output_tokens * \
                    model_info['model_cost']['output_token_cost']

        # def on_llm_new_token(self, token, **kwargs):
        #     Says I need streaming enabled for this to work. maybe later
        #     self.input_tokens += 1

    # Create the LLM model
    def __get_llm_model(self):
        if self.__log_verbose:
            print("RagChatbot: Initializing LLM model")
            print(f"Creating LLM model: {self.__current_model['name']}")
            print(f"  - client: {self.__current_model['client_name']}")
            print(f"  - id: {self.__current_model['id']}")

        llm_model_kwargs = self.__current_model['kwargs']

        # Override settings with the prompt model settings
        llm_model_kwargs['temperature'] = self.__prompt_model.llm_temperature
        if 'topP' in llm_model_kwargs:
            llm_model_kwargs['topP'] = self.__prompt_model.llm_top_p

        if 'top_p' in llm_model_kwargs:
            llm_model_kwargs['top_p'] = self.__prompt_model.llm_top_p

        llm_model = None
        # Handle creating Bedrock Models
        if self.__current_model['client_name'] == 'bedrock-runtime':
            # Explicitly create client with boto3 to get better control and transparency
            bedrock = boto3.client('bedrock-runtime', region_name=self.__current_model['region_name'])

            if self.__current_model['key'] == LlmModelTypes.BEDROCK_CLAUDE3_SONNET:
                llm_model = BedrockChat(
                    model_id=self.__current_model['id'],
                    client=bedrock,
                    model_kwargs=llm_model_kwargs
                )

            else:
                llm_model = Bedrock(
                    model_id=self.__current_model['id'],
                    client=bedrock,
                    model_kwargs=llm_model_kwargs
                )

        elif self.__current_model['client_name'] == 'openai':
            temp = llm_model_kwargs['temperature']
            llm_model = ChatOpenAI(model=self.__current_model['id'], temperature=temp)

        elif self.__current_model['client_name'] == 'google':
            temp = llm_model_kwargs['temperature']
            llm_model = ChatGoogleGenerativeAI(model=self.__current_model['id']
                                               , temperature=temp
                                               , convert_system_message_to_human=True)


        else:
            raise Exception("Unsupported client_name: {}".format(self.__current_model['client_name']))

        return llm_model

    def get_llm_model(self):
        return self.__llm_model

    def restate_question_from_history(self, question, conversation_history):
        summary_prompt = ChatPromptTemplate.from_template(self.__prompt_model.summary_prompt_template)
        history = ""
        for qa_pair in conversation_history:
            history += f"Question - {qa_pair['question']}\nResponse - {qa_pair['response']}\n"
        summary_request = summary_prompt.format(chat_history=history, question=question)
        restated_question = self.__llm_model.invoke(summary_request)
        return restated_question

    def ask_question(self, question, conversation_history=None, verbose=False):
        my_callback_handler = self.RagCallback(self.__llm_model, verbose=verbose)
        chain_config = RunnableConfig(callbacks=[my_callback_handler])

        # --- If we have history, then summarize along with the question to produce a new question ---
        # format of chat history is: [{"question": "****", "response": "****"},]
        #  and is managed and formulated by the client to this call
        restated_question = None
        if conversation_history is not None:
            restated_question = self.restate_question_from_history(question, conversation_history)
            # summary_prompt = ChatPromptTemplate.from_template(self.__prompt_model.summary_prompt_template)
            # history = ""
            # for qa_pair in conversation_history:
            #     history += f"Question - {qa_pair['question']}\nResponse - {qa_pair['response']}\n"
            # summary_request = summary_prompt.format(chat_history=history, question=question)
            # restated_question = self.__llm_model.invoke(summary_request, config=chain_config)

        else:
            restated_question = question

        # --- Now we have a restated question, we ask the with the intended prompts ---
        # Get a retriever
        retriever = self.__vector_db.as_retriever()

        # Register my callback handler

        template = self.__prompt_model.system_prompt_template
        prompt = ChatPromptTemplate.from_template(template)

        # This takes the context from the retriever, based on the question, then
        #   passes the resulting context and question to the prompt template
        #   then executes the query and returns.
        my_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.__llm_model
                | StrOutputParser()
        )
        # provide config to the chain, which at the moment includes the callback handler
        my_chain = my_chain.with_config(chain_config)
        start_time = datetime.now()
        if type(restated_question) == AIMessage:
            answer = my_chain.invoke(restated_question.content)
        else:
            answer = my_chain.invoke(restated_question)

        end_time = datetime.now()
        duration = end_time - start_time

        self.__last_run_input_tokens = my_callback_handler.input_tokens
        self.__last_run_output_tokens = my_callback_handler.output_tokens
        self.__last_run_cost = my_callback_handler.cost(self.__current_model)
        self.__total_cost += my_callback_handler.cost(self.__current_model)
        self.__total_input_tokens += my_callback_handler.input_tokens
        self.__total_output_tokens += my_callback_handler.output_tokens
        self.__last_run_prompt = my_callback_handler.prompt
        self.__last_run_duration = duration.total_seconds()
        self.__total_run_duration += duration.total_seconds()

        # if type(response) == AIMessage:
        #     return response
        # else:  # This is needed because some LLM's will return a string instead of a AIMessage, like Titan Express
        #     return AIMessage(content=response)

        if type(answer) == AIMessage:
            return answer.content
        else:
            return answer

    def get_model_run_summary(self):
        return {
            "total_cost": self.__total_cost,
            "total_input_tokens": self.__total_input_tokens,
            "total_output_tokens": self.__total_output_tokens,
            "total_run_duration": self.__total_run_duration,

            "last_run_input_tokens": self.__last_run_input_tokens,
            "last_run_output_tokens": self.__last_run_output_tokens,
            "last_run_cost": self.__last_run_cost,
            "last_run_duration": self.__last_run_duration,

            "model_name": self.__current_model,
            "vector_db_index": self.__prompt_model.llm_index
        }

    def get_last_run_prompt(self):
        return self.__last_run_prompt

    def get_vector_db(self):
        return self.__vector_db
