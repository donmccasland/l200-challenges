import getpass
import os
import random
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

llm = ChatVertexAI(model="gemini-1.5-flash")

import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
PROJECT_ID="donmac-l200-proj-2"
LOCATION="us-central1"
import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

embeddings = VertexAIEmbeddings(model_name="text-embedding-004")


# 1. Load, chunk and index the contents of the blog to create a retriever.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=embeddings
)
retriever = vectorstore.as_retriever()


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an expert in travel and arranging travel plans."
    "Please help the customer with information about travelling,"
    "by answering their questions directly."
    "If you don't know the answer, say that you"
    "don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []


class SimpleChatBot:
    def __init__(self):
        self.greetings = [
            "Hello!",
            "Hi there!",
            "How are you today?"
        ]
        self.responses = [
            "Tell me more!",
            "Interesting.",
            "I see.",
            "Can you elaborate?"
        ]

    def greet(self):
        return random.choice(self.greetings)

    def respond(self, message):
        ai_msg = rag_chain.invoke({"input": message, "chat_history": chat_history})

        chat_history.extend(
            [
                HumanMessage(content=message),
                AIMessage(content=ai_msg["answer"]),
            ]
        )
        return ai_msg["answer"]


if __name__ == "__main__":
    chatbot = SimpleChatBot()
    print(chatbot.greet())

    while True:
        message = input("Human: ")
        if message.lower() == "quit":
            break
        print("Bot:", chatbot.respond(message))
