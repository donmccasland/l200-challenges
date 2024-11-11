#!/usr/bin/env python3

from google.cloud import aiplatform
import pprint

def connect_prediction_client(
    project: str,
    endpoint_id: str,
    location: str = "us-west1",
    api_endpoint: str = "us-west1-aiplatform.googleapis.com",
):
    """
    Establish connection with prediction client and return client object
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    return client

def prompt_prediction_service(
    client,
    question: str,
    project: str,
    endpoint_id: str,
    location: str = "us-west1",
    api_endpoint: str = "us-west1-aiplatform.googleapis.com",
):
    """
    Convert message from user into prompt, query prediction service and return responses
    """
    # The format of each instance should conform to the deployed model's prediction input schema.
    prompt = """You are an expert in travel and arranging travel plans.
                Please help the customer with information about travelling, 
                buying airplane and train tickets, 
                scheduling lodging at hotels or airbnbs, 
                and site seeing.  
                This is the log of the chat session so far:
                <start_of_turn>user 
                """ + question + """
                <end_of_turn>
                <start_of_turn>model"""
    instances = {"prompt":prompt,
            "max_tokens":200, 
            "temperature":1.0, 
            "top_p":1.0, 
            "top_k":1}
    print(instances)
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(endpoint=endpoint, instances=[instances])
    return response.predictions[0]


client = connect_prediction_client(
    project="455315963016",
    endpoint_id="783525180092710912",
    location="us-west1",
    )


response = ''
question = 'Please give me the names of all the cities in France.'
response = prompt_prediction_service(client, question=question, project="455315963016", endpoint_id="783525180092710912", location="us-west1")
print(response)
