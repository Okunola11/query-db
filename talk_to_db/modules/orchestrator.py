from typing import List, Optional, Tuple
import autogen

from talk_to_db.modules import llm


class Orchestrator:
    """
    Orchestrates conversations between multiple agents.

    This class facilitates communication between agents, handling message formats, function calls, and conversation flow. It can run conversations sequentially or broadcast messages to all agents.

    Attributes:
        name (str): Name of the Orchestrator.
        agents (List[autogen.ConversableAgent]): List of agents participating in the conversation.
        messages (List): History of messages exchanged during the conversation.
        complete_keyword (str): Keyword signifying successful conversation completion (default: "APPROVED").
        error_keyword (str): Keyword signifying conversation failure (default: "ERROR").
    """

    def __init__(self, name: str, agents: List[autogen.ConversableAgent]):
        """Initializes the Orchestrator with a name and a list of agents.

        Args:
            name (str): Name of the Orchestrator.
            agents (List[autogen.ConversableAgent]): List of conversational agents.

        Raises:
            Exception: If the number of agents is less than 2.
        """

        self.name = name
        self.agents = agents
        self.messages = []
        self.complete_keyword = "APPROVED"
        self.error_keyword = "ERROR"

        if len(self.agents) < 2:
            raise Exception("Orchestrator needs at least two agents")

    @property
    def total_agents(self):
        return len(self.agents)

    @property
    def last_message_is_dict(self):
        return isinstance(self.messages[-1], dict)

    @property
    def last_message_is_string(self):
        return isinstance(self.messages[-1], str)

    @property
    def latest_message(self) -> Optional[str]:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def last_message_is_func_call(self):
        return self.last_message_is_dict and self.latest_message.get("function_call", None)

    @property
    def last_message_is_content(self):
        return self.last_message_is_dict and self.latest_message.get("content", None)

    def get_messages_as_string(self):
        """Concatenates all messages in the conversation history into a single string.

        Returns:
            str: Concatenated string representation of all messages.
        """

        messages_as_str = ""

        for message in self.messages:
            if message is None:
                continue

            if isinstance(message, dict):
                content_from_dict = message.get("content", None)
                func_call_from_dict = message.get("function_call", None)
                content = content_from_dict or func_call_from_dict
                if not content:
                    continue
                messages_as_str += str(content)
            else:
                messages_as_str += str(message)

        return messages_as_str

    def get_cost_and_tokens(self):
        """Estimates the cost and number of tokens required to process the conversation history.

        Uses an external `llm.estimate_price_and_tokens` function (assumed to be defined elsewhere)

        Returns:
            Tuple[float, int]: Estimated cost and number of tokens.
        """

        return llm.estimate_price_and_tokens(self.get_messages_as_string())

    def add_message(self, message: str):
        """Appends a message to the conversation history.

        Args:
            message (str): The message to add.
        """

        self.messages.append(message)

    def has_functions(self, agent: autogen.ConversableAgent):
        """Checks if the provided agent has any registered functions.

        Args:
            agent (autogen.ConversableAgent): The agent to check.

        Returns:
            bool: True if the agent has functions, False otherwise.
        """

        return len(agent._function_map) > 0

    def basic_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str
    ):
        """Conducts a basic conversation exchange between two agents.

        Sends a message from `agent_a` to `agent_b`, retrieves a reply, and adds it to the conversation history.

        Args:
            agent_a (autogen.ConversableAgent): The sending agent.
            agent_b (autogen.ConversableAgent): The receiving agent.
            message (str): The message to send.
        """

        print(f"basic_chat(): '{agent_a.name}' --> '{agent_b.name}'")

        agent_a.send(message, agent_b)

        reply = agent_b.generate_reply(sender=agent_a)

        self.add_message(reply)

        print(f"basic_chat(): replied with:", reply)

    def memory_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str
    ):
        """Conducts a conversation with memory retention for the receiving agent.

        Sends a message from `agent_a` to `agent_b`, then sends the reply from `agent_b` back to itself.
        This allows the receiving agent to potentially utilize the conversation history in its next response.

        Args:
            agent_a (autogen.ConversableAgent): The sending agent.
            agent_b (autogen.ConversableAgent): The receiving agent.
            message (str): The message to send.
        """

        print(f"memory_chat() '{agent_a.name}' --> '{agent_b.name}'")

        agent_a.send(message, agent_b)

        reply = agent_b.generate_reply(sender=agent_a)

        agent_b.send(reply, agent_b)

        self.add_message(reply)

    def function_chat(
        self,
        agent_a: autogen.ConversableAgent,
        agent_b: autogen.ConversableAgent,
        message: str
    ):
        """Conducts a conversation involving a function call from the sending agent.

        1. Performs a basic chat with `agent_a` to elicit a function call as a message.
        2. Checks if the function call is valid and can be executed by `agent_a`.
        3. If the function call is valid, executes the function and sends the result to `agent_b`.

        Args:
            agent_a (autogen.ConversableAgent): The sending agent.
            agent_b (autogen.ConversableAgent): The receiving agent.
            message (str): The message to send.
        """

        print(f"function_chat(): '{agent_a}' --> '{agent_b}'")
        
        self.basic_chat(agent_a, agent_a, message)

        assert self.last_message_is_content

        self.basic_chat(agent_a, agent_b, self.latest_message)

    def sequential_conversation(self, prompt) -> Tuple[bool, List[str]]:
        """
        Runs a sequential conversation between agents, passing the prompt from one agent to the next in a chain.

        For example
            "Agent A" -> "Agent B" -> "Agent C" -> "Agent D" -> "Agent E"

        Args:
            prompt (str): The initial prompt for the conversation.

        Returns:
            Tuple[bool, List[str]]: A tuple containing whether the conversation was successful and the conversation history.
        """

        print(f"\n\n---------- {self.name} Orchestrator Starting----------\n\n")

        self.add_message(prompt)

        for idx, agent in enumerate(self.agents):
            agent_a = self.agents[idx]
            agent_b = self.agents[idx + 1]

            print(
                f"\n\n---------- Running iteration {idx} with (agent_a: {agent_a.name}), (agent_b: {agent_b.name}) ----------\n\n"
            )

            # agent_a -> chat -> agent_b
            if self.last_message_is_string:
                self.basic_chat(agent_a, agent_b, self.latest_message)

            # agent_a -> func() -> agent_b
            if self.last_message_is_func_call and self.has_functions(agent_a):
                self.function_chat(agent_a, agent_b, self.latest_message)

            # Ending the process
            if idx == self.total_agents - 2:
                print(f"---------- Orchestrator Complete ----------\n\n")

                was_successful = self.complete_keyword in self.latest_message

                if was_successful:
                    print("✅ Orchestrator was successful")
                else:
                    print("❌ Orchestrator failed")

                return was_successful, self.messages

    def broadcast_conversation(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Broadcasts a prompt to all agents and collects their responses.

        For example:
            "Agent A" -> "Agent B"
            "Agent A" -> "Agent C"
            "Agent A" -> "Agent D"
            "Agent A" -> "Agent E"

        Args:
            prompt (str): The prompt to broadcast.

        Returns:
            Tuple[bool, List[str]]: A tuple containing whether the broadcast was successful and the conversation history.
        """

        print(f"\n\n---------- {self.name} Orchestrator Starting ----------\n\n")

        self.add_message(prompt)

        broadcast_agent = self.agents[0]

        for idx, agent_iterate in enumerate(self.agents[1:]):
            print(f"\n\n---------- Running iteration {idx} with (broadcast_agent: {broadcast_agent.name}, agent_iteration: {agent_iterate.name}) ----------\n\n")

            # agent_a -> chat -> agent_b
            if self.last_message_is_string:
                self.memory_chat(broadcast_agent, agent_iterate, prompt)

            # agent_a -> func() -> agent_b
            if self.last_message_is_func_call and self.has_functions(agent_iterate):
                self.function_chat(agent_iterate, agent_iterate, self.latest_message)

        print(f"---------- Orchestrator Complete ----------\n\n")

        print(f"✅ Orchestrator was successful")

        return True, self.messages
