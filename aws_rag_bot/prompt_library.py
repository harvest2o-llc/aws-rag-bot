from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate


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


# class ClaudNasaSpokespersonPrompts:
#     prompt_model_name = "Claude NASA Spokesperson"
#     llm_temperature = 0.5
#     llm_top_p = 0.5
#     llm_index = "index-artemis-mission"
#     contextualize_q_system_prompt = """Given a chat history and the latest user question \
#         which might reference context in the chat history, formulate a standalone question \
#         which can be understood without the chat history. Do NOT answer the question, \
#         just reformulate it if needed and otherwise return it as is."""
#
#     qa_system_prompt = """You are the spokesperson for NASA speaking to a  \
#     group of journalists.  Respond in a way that is informative and engaging. \
#     Use the following pieces of retrieved context to answer the question. \
#     keep the answer concise.\
#
#     {context}"""
#
#     task_content = """
#     You are the spokesperson for NASA named Sally speaking to a  \
#         group of journalists about the Artemis space program. \
#         \
#         keep the answer concise.\
#     """
#
#     tone_and_context = """Keep the tone friendly, positive, informative and engaging. """
#
#     rules_and_guidelines = """
#     If the question is not related to NASA or the Artemis program,
#     just say I am here to talk about NASA and our mission to the moon."""
#
#     examples = """"""
#
#     # History goes here
#     conversation_history_summary = """"""
#
#     # retrieval document chunks go here
#     supporting_documents = "{context}"
#
#     # Ask for proof of reference to help
#     proof = """Please provide the most relevant sentences that support your answer inside <proof> tags."""
#
#     # Then the question :)
#     human_question = "{question}"
#
#     # Bring it all together
#     total_prompt = f"\n\nHuman: {task_content}\n{tone_and_context}\n{rules_and_guidelines}\n{examples}\n{conversation_history_summary}\n{supporting_documents}\n{proof}\n{human_question}\nAssistant:"
#
#     claude_qa_prompt = f"""
#     {task_content}\n
#     {tone_and_context}\n
#     {rules_and_guidelines}\n
#     {examples}\n
#     {conversation_history_summary}\n
#     {supporting_documents}\n
#     {proof}\n
#     {human_question}\n
#     """
#
#
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{question}"),
#         ]
#     )
#
#     # qa_prompt = ChatPromptTemplate.from_messages(
#     #     [
#     #         ("system", qa_system_prompt),
#     #         MessagesPlaceholder(variable_name="chat_history"),
#     #         ("human", "{question}"),
#     #     ]
#     # )
#
#     qa_prompt2 = PromptTemplate.from_messages(
#         [
#             ("human", qa_system_prompt),
#             ("system", qa_system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{question}"),
#         ]
#     )



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

