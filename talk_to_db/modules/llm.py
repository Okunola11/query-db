"""
Purpose:
    Interact with the OpenAI API.
    Provide supporting prompt engineering functions.
"""

import sys
import openai
from typing import Any, Dict

from talk_to_db.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# -------------------- helpers --------------------

def safe_get(data, dot_chained_keys):
    """Safely retrieves a value from a nested dictionary or list using dot-chained keys.

    Args:
        data: The dictionary or list to access.
        dot_chained_keys: A string representing the dot-chained keys to follow.

    Returns:
        The value at the specified path, or None if it doesn't exist or an error occurs.
    """

    keys = dot_chained_keys.split(".")
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[int(key)]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None

    return data

def response_parse(response: Dict[str, Any]):
    """Extracts the content from the first choice message in a response.

    Args:
        response: A dictionary representing the response.

    Returns:
        The content of the first choice message, or None if it doesn't exist.
    """

    return safe_get(response, "choices.0.message.content")


# -------------------- content generators --------------------

def prompt(prompt: str, model: str = "gpt-4o-mini", instructions: str = "You are a helpful assistant.") -> str:
    """Generates a response to a prompt using the OpenAI API.

    Args:
        prompt: The prompt string to use.
        model: The OpenAI model to use (optional, defaults to "gpt-4o-mini").
        instructions: The system instructions as a message

    Returns:
            The generated response as a string, or None if an error occurs.

    Raises:
        SystemExit: If the OpenAI API key is not set.
    """

    # validate the openai api key - if it's not valid, raise an error
    if not openai.api_key:
        sys.exit(
            """
            ERROR: OpenAI Key not found. Please export your key to OPENAI_API_KEY
            Example bash command:
                export OPENAI_API_KEY=<your openai api key>
            """
        )
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    
    print("--------- OPEN AI RESPONSE ---------")
    print(response.json())
    return response_parse(response.model_dump())


def add_cap_ref(prompt: str, prompt_suffix: str, cap_ref: str, cap_ref_content: str) -> str:
    """
    Attaches a capitalized reference to the prompt.

    Example:
        prompt = 'Refactor this code.'
        prompt_suffix = 'Make it more readable using this EXAMPLE.'
        cap_ref = 'EXAMPLE'
        cap_ref_content = 'def foo():\n     return True'
    
    Returns:
        'Refactor this code. Make it more readable using this Example.\n\nEXAMPLE\n\ndef foo():\n       return True'
    """

    new_prompt = f"""{prompt} {prompt_suffix}\n\n{cap_ref}\n\n{cap_ref_content}"""

    return new_prompt