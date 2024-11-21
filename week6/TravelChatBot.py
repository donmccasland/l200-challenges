import os
import random
from langchain_google_vertexai import (
        ChatVertexAI,
        VertexAIEmbeddings,
        HarmBlockThreshold,
        HarmCategory
)
from langchain.chains import (
        create_retrieval_chain, 
        create_history_aware_retriever, 
        create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
import vertexai
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

class TravelChatBot:
    def __init__(self):
        ### Model init
        PROJECT_ID="donmac-l200-proj-2"
        LOCATION="us-central1"
        # Initilialize safety filters for vertex model
        # This is important to ensure no evaluation responses are blocked
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.llm = ChatVertexAI(model="gemini-1.5-flash")

        ### Vector store init
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        loader = JSONLoader(file_path="./hotels.json", jq_schema=".hotels[]", text_content=False)
        docs = loader.load()

        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=docs, embedding=embeddings
        )
        self.retriever = self.vectorstore.as_retriever()


        ### Prompt templates
        #Basic travel focused prompt template
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

        #Question rewriter prompt
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
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)


        self.chat_history = []
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

    def model(self):
        return self.llm

    def greet(self):
        return random.choice(self.greetings)

    def respond(self, message):
        ai_msg = self.rag_chain.invoke({"input": message, "chat_history": self.chat_history})

        self.chat_history.extend(
            [
                HumanMessage(content=message),
                AIMessage(content=ai_msg["answer"]),
            ]
        )
        return ai_msg["answer"]

