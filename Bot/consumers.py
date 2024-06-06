# import os
# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# from langchain_chroma import Chroma
# # from dotenv import load_dotenv
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# from asgiref.sync import sync_to_async
# # import torch
#
# # Load environment variables from .env file
# # load_dotenv()
#
# # Initialize the model and tokenizer variables
# model = None
# tokenizer = None
#
# def initialize_model():
#     pass
# #     global model, tokenizer
# #     model_name = "desaitrushti/Llama-2-7b-chat-finetune"  # Use a smaller model for testing purposes
# #     # model_name = "distilgpt2"
# #     try:
# #         tokenizer = AutoTokenizer.from_pretrained(model_name)
# #         print("TOkenizer")
# #         model = AutoModelForCausalLM.from_pretrained(model_name)
# #         print("Model and tokenizer loaded successfully.")
# #     except Exception as e:
# #         print(f"Error loading model and tokenizer: {e}")
# #         raise
#
# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         print("WebSocket connection opened")
#         await self.accept()
#
#     async def disconnect(self, close_code):
#         print("WebSocket connection closed with code:", close_code)
#
#     async def receive(self, text_data):
#         print("Received message:", text_data)
#         text_data_json = json.loads(text_data)
#         message = text_data_json['message']
#
#         print("Before Call")
#         response_message = await self.process_user_query(message)
#         print("After Call")
#         print("Sending response:", response_message)
#
#         # Send back the response
#         await self.send(text_data=json.dumps({
#             'message': response_message
#         }))
#
#     @sync_to_async
#     def process_user_query(self, query):
#         chroma_directory = r'D:\IndiaNIC\chatbot\db'
#         db = Chroma(persist_directory=chroma_directory)
#
#         print(db)
#
#         # global model, tokenizer
#         # if model is None or tokenizer is None:
#         #     initialize_model()
#         # try:
#         #     # Generate content using the Hugging Face model
#         #     inputs = tokenizer.encode(query, return_tensors="pt")
#         #     outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
#         #     ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         #     return ai_response
#         # except Exception as e:
#         #     print(f"Error generating content: {e}")
#         #     return "Error generating content"
#
#
# if __name__ == "__main__":
#     # For debugging, run this section to see if the model loads correctly
#     initialize_model()
#     print("Initialization complete. Model is ready for use.")
import os
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer
from asgiref.sync import sync_to_async
from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

# from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# Load environment variables from .env file


chat_history=[]
class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        global chat_history
        chat_history = []
        print("WebSocket connection opened")
        await self.accept()

    async def disconnect(self, close_code):
        print("WebSocket connection closed with code:", close_code)

    async def receive(self, text_data):
        print("Received message:", text_data)
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        # Process the user query
        response_message = await self.process_user_query(message)

        print("Sending response:", response_message)

        # Send back the response
        await self.send(text_data=json.dumps({
            'message': response_message
        }))

    @sync_to_async
    def process_user_query(self, query):
        # chat_history = []
        global chat_history
        if len(chat_history) > 11:
            chat_history = []
        print(len(chat_history))
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        chroma_directory = r'D:\IndiaNIC\chatbot\db'
        db = Chroma(persist_directory=chroma_directory, embedding_function=hf)

        # Initialize the Chroma vector store with Google AI embeddings
        retriever = db.as_retriever()
        # print("Db connected")

        google_api_key = "AIzaSyAVDeoinwTOBEJD6RTzcrbd3GMn0hsKglI"
        # if not google_api_key:
        #     raise ValueError("Google API key not found in environment variables.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        qa_system_prompt = """You are a chatbot ai assistant at indiaNIC.
            act as a humanoid chatbot and give responses such that you are employed at indiaNIC.
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\
            {context}"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # prompt = ChatPromptTemplate.from_template("""
        # Act as a chatbot of IndiaNIC Infotech company.
        # Answer the following question based only on the provided context.
        # Provide a detailed, short, and sweet answer as a chatbot which is helpful.
        # <context>
        # {context}
        # </context>
        # Question: {input} at indiaNIC
        # """)

        document_chain = create_stuff_documents_chain(llm, prompt, output_parser=StrOutputParser())
        # document_chain = create_stuff_documents_chain(llm, prompt, output_parser=StrOutputParser())
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
        # retrieval_chain = create_retrieval_chain(retriever, document_chain)
        # memory=ConversationBufferMemory()
        # print(chat_history,"*************************")
        response = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
        # response = retrieval_chain.invoke({"input": query, "context": memory})
        # memory.add(query,response["answer"])
        chat_history.extend([HumanMessage(content=query), response["answer"]])
        # print(chat_history,"#############################")
        return response['answer']
