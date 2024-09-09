"""
Purpose:
    Interact with the OpenAI API.
    Provide supporting prompt engineering functions.
"""

import sys
import openai
from typing import Any, Dict
import tiktoken
from typing import List
import json

from talk_to_db.types import TurboTool
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

def prompt_func(
    prompt: str,
    turbo_tools: List[TurboTool],
    model: str = "gpt-4o-mini",
    instructions: str = "You are a helpful assistant"
) -> str:
    """
    Processes a prompt using a specified AI model and a list of TurboTools to generate and handle tool-based responses.

    This function sends a prompt to an AI model, incorporates any necessary instructions, and manages responses 
    involving tool calls. It interacts with the TurboTools to execute functions based on the AI model's responses 
    and collects the results.

    Args:
        prompt (str): The user-provided prompt or query that the AI model will process.
        turbo_tools (List[TurboTool]): A list of `TurboTool` objects that provide configurations and functions 
                                        to be used for processing tool-based responses.
        model (str, optional): The AI model to use for generating responses. Default is "gpt-4o-mini".
        instructions (str, optional): Instructions to be provided to the AI model as a system message. 
                                       Default is "You are a helpful assistant".

    Returns:
        List[str]: A list of responses from the tools, which are derived from the AI model's interactions 
                   with the provided `TurboTools`.

    Process:
        1. Creates a list of messages starting with the user prompt.
        2. Adds system instructions to the beginning of the messages.
        3. Configures tool choices based on the number of TurboTools provided.
        4. Sends the prompt and tool configurations to the AI model to get a response.
        5. Processes any tool calls specified in the AI model's response:
            - Matches the tool calls with the provided TurboTools.
            - Executes the tool functions with the provided arguments.
            - Appends responses from the tool functions to the messages list.
        6. Returns the collected responses from the tool functions.

    Example:
        >>> tools = [TurboTool(name="example_tool", config={"type": "function"}, function=some_function)]
        >>> responses = prompt_func("Describe the impact of climate change.", tools)
        >>> print(responses)
        ["Function output from example_tool"]

    Notes:
        - Ensure that the `TurboTool` objects are properly configured and their functions can handle the arguments 
          provided by the AI model.
        - The AI model's response will include tool calls, which are matched and executed accordingly.
    """

    messages = [{"role": "user", "content": prompt}]
    tools = [turbo_tool.config for turbo_tool in turbo_tools]

    tool_choice = (
        "auto"
        if len(turbo_tools) > 1
        else {"type": "function", "function": {"name": turbo_tools[0].name}}
    )

    messages.insert(
        0, {"role": "system", "content": instructions}
    ) # Insert instructions as the first system message

    response = openai.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    func_responses = []

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            for turbo_tool in  turbo_tools:
                if tool_call.function.name == turbo_tool.name:
                    function_response = turbo_tool.function(**json.loads(tool_call.function.arguments))

                    func_responses.append(function_response)

                    message_to_append  = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": turbo_tool.name,
                        "content": function_response
                    }
                    messages.append(message_to_append)

    return func_responses


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

# ------------------- TOKEN COST -------------------

def count_tokens(text: str):
    """Counts the number of tokens in the given text.

    Args:
        text (str): The input text to count tokens from.

    Returns:
        int: The number of tokens in the text.
    """

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def estimate_price_and_tokens(text):
    """Estimates the price and token count for the given text.

    Args:
        text (str): The input text to estimate price and tokens for.

    Returns:
        tuple: A tuple containing the estimated cost (float) and token count (int).
    """

    # round up to the input tokens
    COST_PER_1K_TOKENS = 0.06

    tokens = count_tokens(text)

    estimated_cost = (tokens / 1000) * COST_PER_1K_TOKENS

    # round
    estimated_cost = round(estimated_cost, 2)

    return estimated_cost, tokens