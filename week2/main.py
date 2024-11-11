# chat.py
from typing import List, Tuple

import asyncio
import time
from nicegui import context, ui, app
from pathlib import Path
from nicegui.ui import markdown

import time
import json

import vertexai
from vertexai.generative_models import GenerativeModel, Part

import pprint

preamblelist = ["""You are an expert in travel and arranging travel plans.
Please help the customer with information about travelling, 
buying airplane and train tickets, 
scheduling lodging at hotels or airbnbs, 
and site seeing.  
This is the log of the chat session so far:
"""]

def init_model(
    modelname = "gemini-1.0-pro-002",
    projectid = "donmac-l200-proj-2",
    location = "us-central1"
    ):
    """
    Init vertex ai and return model
    """
    vertexai.init(project=projectid, location=location)

    model = GenerativeModel(modelname)

    return model

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
    model,
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
    response = model.generate_content(prompt)
    updateresponse = """<start_of_turn>model
""" + response.text
    chathistory[-1]["response"] = updateresponse
    print("RESPONSE:\n" + response.text + "\n")
    return response.text


app.add_static_files('/static', Path(__file__).parent.joinpath('static'))
user_avatar = 'static/user1.png'
bot1_avatar = 'static/bot1.png'

model = init_model()

chathistory = []


async def llm(messages, chat_messages, client_id, query):
    url = "http://127.0.0.1:8001"
#    response = await asyncio.get_event_loop().run_in_executor(None, post_request_, url)
    response = send_question(model=model, question=query, chathistory=chathistory)
    messages[-1][-1] += f'{response}'
#    for chunk in response.iter_content(decode_unicode=True):
#        messages[-1][-1] += f'{chunk}'
#        print(messages[-1][-1])
#        chat_messages.refresh()


@ui.page('/')
def main():
    messages: List[List[str, str]] = []
    thinking: bool = False

    @ui.refreshable
    def chat_messages() -> None:
        for name, text in messages:
            ui.chat_message(text=text, name=name, sent=name == 'You',
                            avatar=user_avatar if name == 'You' else bot1_avatar )
        if thinking:
            ui.spinner(size='3rem').classes('self-center')
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

    async def send() -> None:
        nonlocal thinking
        message = text.value
        messages.append(['You', text.value])
        thinking = True
        text.value = ''
        chat_messages.refresh()

        messages.append(['bot', ""])
        thinking = False
        chat_messages.refresh()
        await llm(messages, chat_messages, context.get_client().id, messages[-2][-1])
        chat_messages.refresh()

    anchor_style = r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}'
    ui.timer(1, chat_messages.refresh)
    ui.add_head_html(f'<style>{anchor_style}</style>')

    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')

    with ui.tabs().classes('w-full') as tabs:
        chat_tab = ui.tab('chat')
    with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
        with ui.tab_panel(chat_tab).classes('items-stretch'):
            chat_messages()

    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = ''
            text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)


ui.run()

def foodmain():
    model = init_model()

    chathistory = []
    async def send() -> None:
        question = text.value
        text.value = ''

        with message_container:
            ui.chat_message(text=question, name='You', sent=True)
            response_message = ui.chat_message(name='Bot', sent=False)
            spinner = ui.spinner(type='dots')
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

        reponse = ''
        response = send_question(model=model, question=question, chathistory=chathistory)
        response_message.clear()
        with response_message:
            print("RESPONSE TWO:"+response)
            ui.markdown(response)
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        message_container.remove(spinner)

    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')

    # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')

    with ui.tabs().classes('w-full') as tabs:
        chat_tab = ui.tab('Chat')
    with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
        message_container = ui.tab_panel(chat_tab).classes('items-stretch')
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = 'message'
            text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
        ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
            .classes('text-xs self-end mr-8 m-[-1em] text-primary')

