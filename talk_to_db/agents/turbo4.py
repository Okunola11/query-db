import json
import openai
import time
from openai.types.beta.threads.message import Message
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple, Optional

from talk_to_db.modules import llm
from talk_to_db.types import Chat, TurboTool
from talk_to_db.settings import OPENAI_API_KEY

class Turbo4:
    """A class for managing interactions with OpenAI's GPT-4 Assistant APIs.

    This class provides functionality for creating and managing assistants, threads, and tools,
    as well as utilities for validation, monitoring costs, and saving conversation history.

    Attributes:
        client (openai.OpenAI): The OpenAI client instance.
        map_function_tools (Dict[str, TurboTool]): A dictionary mapping tool names to TurboTool instances.
        current_thread_id (str): The ID of the current thread.
        thread_messages (List[Message]): A list of messages in the current thread.
        local_messages (List[str]): A list of local messages.
        assistant_id (str): The ID of the assistant.
        polling_interval (float): The interval in seconds to poll the API for thread run completion.
        model (str): The model used by the assistant.
    """

    def __init__(self):
        """Initializes the Turbo4 instance with default settings.

        Sets up the OpenAI API key, initializes the OpenAI client, and prepares internal storage
        for thread messages, tools, and assistant information.
        """

        openai.api_key = OPENAI_API_KEY
        self.client = openai.OpenAI()
        self.map_function_tools: Dict[str, TurboTool] = {}
        self.current_thread_id = None
        self.thread_messages: List[Message] = []
        self.local_messages = []
        self.assistant_id = None
        self.polling_interval = 1 # interval in seconds to poll the API for thread run completion
        self.model = "gpt-4o-mini"

    @property
    def chat_messages(self) -> List[Chat]:
        """Gets the list of chat messages from the current thread.

        Converts thread messages into a list of `Chat` objects with information about the sender,
        recipient, message content, and creation time.

        Returns:
            List[Chat]: A list of `Chat` objects representing the messages in the current thread.
        """

        return [
            Chat(
                from_name=msg.role,
                to_name="assistant" if msg.role == "user" else "user",
                message=llm.safe_get(msg.model_dump(), "content.0.text.value"),
                created=msg.created_at
            )
            for msg in self.thread_messages
        ]

    @property
    def tool_config(self):
        """Gets the configuration of all tools equipped on the assistant.

        Returns:
            List[Dict]: A list of tool configurations as dictionaries.
        """

        return [tool.config for tool in self.map_function_tools.values()]

    # ---------------- ADDITIONAL UTILITY FUNCTIONS ----------------

    def run_validation(self, validation_func: Callable):
        """Runs a validation function.

        Args:
            validation_func (Callable): The validation function to execute.

        Returns:
            Turbo4: The current instance for chaining.
        """

        print(f"run_validation({validation_func.__name__})")
        validation_func()
        return self

    def spy_on_assistant(self, output_file: str):
        """Saves the conversation history of the assistant to a JSON file.

        Args:
            output_file (str): The path to the file where conversation history will be saved.

        Returns:
            Turbo4: The current instance for chaining.
        """

        sorted_messages = sorted(
            self.chat_messages, key=lambda msg: msg.created, reverse=False
        )
        messages_as_json = [asdict(msg) for msg in sorted_messages]
        with open(output_file, "w") as f:
            json.dump(messages_as_json, f, indent=2)

        return self

    def get_cost_and_tokens(self, output_file: str) -> Tuple[float, float]:
        """Estimates the cost and token usage of the current thread's messages and saves it to a JSON file.

        Args:
            output_file (str): The path to the file where cost and token information will be saved.

        Returns:
            Tuple[float, float]: A tuple containing the estimated cost and token count.
        """

        retrieval_costs = 0
        code_interpreter_cost = 0

        msgs = [
            llm.safe_get(msg.model_dump(), "content.0.text.value")
            for msg in self.thread_messages
        ]
        joined_msg = " ".join(msgs)

        msg_cost, tokens = llm.estimate_price_and_tokens(joined_msg)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "cost": msg_cost,
                    "tokens": tokens
                },
                f,
                indent=2
            )

        return self

    # ---------------- CORE ASSISTANTS API FUNCTIONS ----------------

    def get_or_create_assistant(self, name: str, model: str = "gpt-4o-mini"):
        """Retrieves an existing assistant by name or creates a new one if it doesn't exist.

        Args:
            name (str): The name of the assistant.
            model (str): The model to use for the assistant (default is "gpt-4o-mini").

        Returns:
            Turbo4: The current instance for chaining.
        """

        print(f"get_or_create_assistant({name}, {model})")

        # Retrieve the list of existing assistants
        assistants: List[Assistant] = self.client.beta.assistants.list().data

        # check if an assistant with the given name already exists
        found = False
        for assistant in assistants:
            if assistant.name == name:
                self.assistant_id = assistant.id
                # update model if different
                if assistant.model != model:
                    print(f"Updating assistant  model from '{assistant.model}' to '{model}'")
                    print(self.assistant_id)
                    self.client.beta.assistants.update(
                        assistant_id=self.assistant_id, model=model
                    )
                if self.assistant_id is None:
                    self.assistant_id = assistant.id
                found = True
                break
        if not found:
            assistant = self.client.beta.assistants.create(model=model, name=name)
            self.assistant_id = assistant.id

        self.model = model

        return self

    def set_instructions(self, instructions: str):
        """Updates the instructions for the currently active assistant.

        Args:
            instructions (str): The new instructions to set for the assistant.

        Returns:
            Turbo4: The current instance for chaining.

        Raises:
            ValueError: If no assistant has been created or retrieved.
        """

        print(f"set_instruction()")
        print(self.assistant_id)
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )
        # Update the assistant with the new instruction
        updated_assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant_id, instructions=instructions
        )
        return self

    def equip_tools(self, turbo_tools: List[TurboTool], equip_on_assistant: bool = False):
        """Updates the tools equipped on the assistant and optionally updates the assistant with these tools.

        Args:
            turbo_tools (List[TurboTool]): The list of `TurboTool` objects to equip.
            equip_on_assistant (bool): Whether to update the assistant with these tools (default is False).

        Returns:
            Turbo4: The current instance for chaining.

        Raises:
            ValueError: If no assistant has been created or retrieved.
        """

        print(f"equip_tools({turbo_tools}, {equip_on_assistant})")
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )

        # Update the function dictionaries with the new tools
        self.map_function_tools = {tool.name: tool for tool in turbo_tools}

        if equip_on_assistant:
            updated_assistant = self.client.beta.assistants.update(
                assistant_id=self.assistant_id, tools=self.tool_config
            )
        return self

    def make_thread(self):
        """Creates a new thread for conversation and initializes internal storage.

        Returns:
            Turbo4: The current instance for chaining.

        Raises:
            ValueError: If no assistant has been created or retrieved.
        """

        print(f"make_thread()")

        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )

        response = self.client.beta.threads.create()
        self.current_thread_id = response.id
        self.thread_messages = []
        return self

    def add_message(self, message: str, refresh_threads: bool = False):
        """Adds a user message to the current thread and optionally refreshes thread messages.

        Args:
            message (str): The message content to add.
            refresh_threads (bool): Whether to refresh thread messages after adding (default is False).

        Returns:
            Turbo4: The current instance for chaining.
        """

        print(f"add_message({message})")
        self.local_messages.append(message)
        self.client.beta.threads.messages.create(
            thread_id=self.current_thread_id, content=message, role="user"
        )
        if refresh_threads:
            self.load_threads()
        return self

    def load_threads(self):
        """Loads messages from the current thread into internal storage.

        This method updates the `thread_messages` attribute with messages from the current thread.
        """

        self.thread_messages = self.client.beta.threads.messages.list(
            thread_id=self.current_thread_id
        ).data

    def list_steps(self):
        """Lists the steps of the current thread run.

        Returns:
            List[Step]: A list of `Step` objects representing the steps of the current thread run.
        """

        print(f"list_steps()")
        steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.current_thread_id, run_id=self.run_id
        )
        print("Steps", steps)
        return steps

    def run_thread(self, toolbox: Optional[List[str]] = None):
        """Starts running the current thread with optional tools and handles polling for completion.

        Args:
            toolbox (Optional[List[str]]): A list of tool names to use for the thread. If None, no tools are used.

        Returns:
            Turbo4: The current instance for chaining.

        Raises:
            ValueError: If no thread has been created or if no messages have been added.
        """

        print(f"run_thread({toolbox})")

        if self.current_thread_id is None:
            raise ValueError(
                "No thread has been created. Call make_thread() first."
            )
        if self.local_messages == []:
            raise ValueError("No messages have been added to the thread")

        if toolbox is None:
            tools = None
        else: 
            # get tools from toolbox
            print(f"\nToolbox contains: {toolbox}")
            print(f"self.map_function_tools: {self.map_function_tools}")
            tools = [self.map_function_tools[tool_name].config for tool_name in toolbox]

            if len(tools) != len(toolbox):
                raise ValueError(
                    f"Tool not found in toolbox. Toolbox={toolbox}, tools={tools}. Make sure all tools are equipped on the assistant."
                )

        # refresh current thread
        self.load_threads()

        # start the thread running
        run = self.client.beta.threads.runs.create(
            thread_id=self.current_thread_id,
            assistant_id=self.assistant_id,
            tools=tools
        )
        self.run_id = run.id

        # Polling mechanism to wait for thread's completion or required actions
        while True:
            # self.list_steps()

            run_status = self.client.beta.threads.runs.retrieve(
                run_id=self.run_id, thread_id=self.current_thread_id
            )
            print(f"run_status ===> {run_status}")
            if run_status.status == "requires_action":
                tool_outputs: List[ToolOutput] = []
                for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                    tool_function = tool_call.function
                    tool_name = tool_function.name

                    # check if tool_arguments is already a dictionary, if so, proceed directly
                    if isinstance(tool_function.arguments, dict):
                        tool_arguments = tool_function.arguments
                    else:
                        # Assume the arguments are JSON string and parse them
                        tool_arguments = json.loads(tool_function.arguments)

                    print(f"run_thread() Calling {tool_name}({tool_arguments})")

                    # Assuming arguments are passed as a dictionary
                    function_output = self.map_function_tools[tool_name].function(**tool_arguments)

                    tool_outputs.append(
                        ToolOutput(tool_call_id=tool_call.id, output=function_output)
                    )
                
                # submit the tool outputs back to the API
                self.client.beta.threads.runs.submit_tool_outputs(
                    run_id=self.run_id,
                    thread_id=self.current_thread_id,
                    tool_outputs=tool_outputs
                )
            
            elif run_status.status == "completed":
                self.load_threads()
                return self

            time.sleep(self.polling_interval) # Wait a little before polling again

    def enable_retrieval(self):
        """Updates the assistant to enable retrieval functionality.

        Returns:
            Turbo4: The current instance for chaining.

        Raises:
            ValueError: If no assistant has been created or retrieved.
        """

        print(f"enable_retrieval()")
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )

        # Update the assistant with the new list of tools, replacing any existing tools
        updated_assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant_id, tools=[{"type": "retrieval"}]
        )

        return self

    # Future version:
        # Enable code interpreter
        # CRUD files