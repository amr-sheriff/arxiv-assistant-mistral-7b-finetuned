"""
Author: Amr Sherif
Version: 1.0.0
Date: 2024-09-20
Description: This script provides helper classes and functions for interacting with the Together API, including chat
             and completion functionalities. It includes CSS styling for Jupyter notebook outputs.

Project: Arxiv Assistant
Dependencies: requests, IPython
License: MIT License

Change Log:
    - Version 1.0.0: Initial version with Chatbot and Completion classes, and a function to set notebook CSS.
"""

import requests
from IPython.display import HTML, display

def set_css():
    """
    Set the CSS style for the notebook to improve the display of long text.
    """
    display(HTML(
        '''
        <style>
        pre {
            white-space: pre-wrap;
        }
        </style>
        '''
    ))


class Chatbot:
    """
    A class for interacting with the Together API to send and receive chat completions.

    Attributes:
    ----------
    api_key : str
        The API key required for authenticating requests to the Together API.
    api_url : str, optional
        The URL for the Together API endpoint for chat completions (default is "https://api.together.ai/chat/completions").
    model : str, optional
        The model identifier for the chatbot (default is "mistralai/Mistral-7B-Instruct-v0.2").
    history : list
        A list to store the conversation history between the user and the assistant.

    Methods:
    -------
    _create_payload(user_input, system_message=None)
        Internal method to create the payload for the API request.

    send_request(user_input, system_message=None)
        Sends a request to the Together API with the provided user input and optional system message, returning the assistant's response.

    start_new_conversation()
        Resets the conversation history to start a new conversation.
    """
    def __init__(self, api_key, model="mistralai/Mistral-7B-Instruct-v0.2", api_url="https://api.together.ai/chat/completions"):
        """
        Initializes the Chatbot instance with the necessary parameters.

        Parameters:
        ----------
        api_key : str
            The API key for authenticating requests to the Together API.
        model : str, optional
            The model identifier to use for chat completions (default is "mistralai/Mistral-7B-Instruct-v0.2").
        api_url : str, optional
            The URL for the Together API endpoint (default is "https://api.together.ai/chat/completions").
        history : list
            A list to store the conversation history between the user and the assistant.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.history = []

    def _create_payload(self, user_input, system_message=None):
        """
        Creates the payload for the API request based on user input and conversation history.

        Parameters:
        ----------
        user_input : str
            The user's input to be sent to the chatbot.
        system_message : str, optional
            An optional system message to set the context for the conversation.

        Returns:
        -------
        dict
            A dictionary representing the payload to be sent in the API request.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.7,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["[/INST]", "</s>"],
            "repetition_penalty": 1,
            "top_k": 50,
        }
        return payload

    def send_request(self, user_input, system_message=None):
        """
        Sends the user input and optional system message to the Together API and returns the assistant's response.

        Parameters:
        ----------
        user_input : str
            The user's input to be sent to the chatbot.
        system_message : str, optional
            An optional system message to provide context for the assistant.

        Returns:
        -------
        str
            The assistant's response to the user's input.

        Raises:
        -------
        Exception
            If the API request fails, an exception is raised with the status code and error message.
        """
        payload = self._create_payload(user_input, system_message)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            assistant_message = result.get("choices")[0].get("message").get("content")

            # Update the history with the new messages
            if system_message:
                self.history.append({"role": "system", "content": system_message})
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def start_new_conversation(self):
        """
        Resets the conversation history, allowing a new conversation to start fresh.
        """
        self.history = []


class Completion:
    """
    A class for sending completion requests to the Together API.

    Attributes:
    ----------
    api_key : str
        The API key required for authenticating requests to the Together API.
    api_url : str
        The URL for the Together API endpoint for completion tasks.
    model : str
        The model identifier to be used for generating text completions.

    Methods:
    -------
    _create_payload(prompt):
        Creates the payload for the API request based on the provided prompt.

    send_request(prompt):
        Sends a completion request to the Together API with the provided prompt and returns the generated completion.
    """
    def __init__(self, api_key, model="mistralai/Mistral-7B-v0.1", api_url="https://api.together.xyz/v1/completions"):
        """
        Initializes the Completion instance with the required API key, model, and API URL.

        Parameters:
        ----------
        api_key : str
            The API key for authenticating requests to the Together API.
        model : str, optional
            The model identifier to use for text completions (default is "mistralai/Mistral-7B-v0.1").
        api_url : str, optional
            The URL for the Together API completion endpoint (default is "https://api.together.xyz/v1/completions").
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    def _create_payload(self, prompt):
        """
        Creates the payload for the API request based on the provided prompt.

        Parameters:
        ----------
        prompt : str
            The prompt or input text to be sent to the model for generating a completion.

        Returns:
        -------
        dict
            A dictionary representing the payload to be sent in the API request.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.7,
            "stop": ["</s>"],
            "repetition_penalty": 1,
            "top_k": 50,
        }
        return payload

    def send_request(self, prompt):
        """
        Sends a completion request to the Together API with the provided prompt and returns the generated completion.

        Parameters:
        ----------
        prompt : str
            The prompt to be sent to the API for generating a completion.

        Returns:
        -------
        str
            The generated completion text from the model.

        Raises:
        -------
        Exception
            If the API request fails, an exception is raised with the status code and error message.
        """
        payload = self._create_payload(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            completion_text = result.get("choices")[0].get("text")
            return completion_text
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")