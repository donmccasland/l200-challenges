#!/usr/bin/env python3
#from langchain_google_vertexai.gemma import GemmaChatVertexAIModelGarden
#from langchain_core.messages import (
#    AIMessage,
#    HumanMessage,
#)
#from log_callback_handler import NiceGuiLogElementCallbackHandler

from nicegui import ui

from typing import Dict, List, Union

from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

import pprint

preamblelist = ["""You are an expert in travel and arranging travel plans.
Please help the customer with information about travelling, 
buying airplane and train tickets, 
scheduling lodging at hotels or airbnbs, 
and site seeing.  
This is the log of the chat session so far:
"""]

def connect_prediction_client(
    project: str,
    endpoint_id: str,
    location: str = "us-west1",
    api_endpoint: str = "us-west1-aiplatform.googleapis.com",
):
    """
    Establish connection with prediction client and return client object
    """
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)

    return client,endpoint

def build_prompt(question, preamble, chathistory):
    """
    Combine question with prompt preamble and chat history to build up final prompt to chat client
    """
    prompt = preamble 
    question = """<start_of_turn>user
""" + question + """
<end_of_turn>
"""
    stubresponse = """<start_of_turn>model
"""
    chathistory.append({"question":question, "response":stubresponse})
    for entry in chathistory:
        prompt = prompt + entry["question"]
        prompt = prompt + entry["response"]
    return prompt


def send_question(
    client,
    endpoint,
    question: str,
    chathistory,
    max_tokens=128,
    temperature=1.0,
    top_p=1.0,
    top_k=1,
):
    """
    Convert message from user into prompt, query prediction service and return responses
    """
    # The format of each instance should conform to the deployed model's prediction input schema.
    prompt = build_prompt(question, preamblelist[0], chathistory)
    print("PROMPT:\n" + prompt + "\n")
    instances = {"prompt":prompt,
            "max_tokens":max_tokens, 
            "temperature":temperature, 
            "top_p":top_p, 
            "top_k":top_k}
    response = client.predict(endpoint=endpoint, instances=[instances])
    updateresponse = """<start_of_turn>model
""" + response.predictions[0]
    chathistory[-1]["response"] = updateresponse
    print("RESPONSE:\n" + response.predictions[0] + "\n")
    return response.predictions[0]


@ui.page('/')
def main():
#    print("1")
    client,endpoint = connect_prediction_client(
        project="455315963016",
        endpoint_id="783525180092710912",
        location="us-west1",
        )

    chathistory = []
#    print("2")
    async def send() -> None:
#        print("3")
        question = text.value
        text.value = ''

        with message_container:
#            print("4")
            ui.chat_message(text=question, name='You', sent=True)
            response_message = ui.chat_message(name='Bot', sent=False)
            spinner = ui.spinner(type='dots')
#        print("5")
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

        reponse = ''
        response = send_question(client, endpoint=endpoint, question=question, chathistory=chathistory)
#        print("6")
        response_message.clear()
        with response_message:
#            print("7")
            ui.markdown(response)
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        message_container.remove(spinner)

#    print("8")
    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')

    # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')

#    print("9")
    with ui.tabs().classes('w-full') as tabs:
        chat_tab = ui.tab('Chat')
    with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
        message_container = ui.tab_panel(chat_tab).classes('items-stretch')
#    print("10")
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = 'message'
            text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
        ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
            .classes('text-xs self-end mr-8 m-[-1em] text-primary')

#print("11")
ui.run(title='Chat with Google Gemma-2 on Vertex', port=8081)

