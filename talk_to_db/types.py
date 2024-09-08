from dataclasses import dataclass
from typing import List


@dataclass
class Chat:
    """Represents a single chat message.

    This is a model for conversations between the AI agents.

    Attributes:
        from_name (str): The name of the sender of the message.
        to_name (str): The name of the recipient of the message.
        message (str): The content of the chat message.
    """

    from_name: str
    to_name: str
    message: str


@dataclass
class ConversationResult:
    """Represents the result of a conversation between agents.

    This is the model of the results from the AI agents.

    Attributes:
        success (bool): Indicates whether the conversation was successful.
        messages (List[Chat]): A list of chat messages exchanged during the conversation.
        cost (float): An estimate of the cost of the conversation, based on the number of tokens used.
        tokens (int): The number of tokens used during the conversation.
        last_message_str (str): The last message sent in the conversation.
        error_message (str): Error message during the conversation.
    """

    success: bool
    messages: List[Chat]
    cost: float
    tokens: int
    last_message_str: str
    error_message: str