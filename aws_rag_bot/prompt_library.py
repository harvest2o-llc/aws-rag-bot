from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder


class DefaultPrompts:
    prompt_model_name = "Assistant"
    llm_temperature = 0.8
    llm_top_p = 0.5
    llm_index = "index-artemis-mission"
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


class NasaSpokespersonPrompts:
    prompt_model_name = "NASA Spokesperson"
    llm_temperature = 0.5
    llm_top_p = 0.5
    llm_index = "index-artemis-mission"
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    qa_system_prompt = """You are the spokesperson for NASA speaking to a  \
    group of journalists.  Respond in a way that is informative and engaging. \
    Use the following pieces of retrieved context to answer the question. \
    keep the answer concise.\

    {context}"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

