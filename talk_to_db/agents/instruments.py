import os

from talk_to_db.modules.db import PostgresManager
from talk_to_db.modules import file
from talk_to_db.settings import BASE_DIR


class AgentInstruments:
    """
    Base class for multi-agent instruments that are tools, state, and functions that an agent can use across the lifecycle of conversations.

    Attributes:
        session_id (str): The unique identifier for the current session.
        messages (list): A list of messages exchanged during the conversation.

    Methods:
        sync_messages(messages: list): Syncs messages with the orchestrator.
        root_dir: A property that returns the root directory for the session.
        agent_chat_file: A property that returns the path to the agent chat file.
    """

    def __init__(self) -> None:
        self.session_id = None
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def sync_messages(self, messages: list):
        """Syncs messages with the orchestrator.   

        Args:
            messages (list): A list of messages to sync.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        raise NotImplementedError

    def make_agent_chat_file(self, team_name: str):
        """Returns the path to the agents chat file for a team.

        Returns:
            str: The path to the team's agent chat file.
        """

        return os.path.join(self.root_dir, f"agents_chat_{team_name}.json")

    @property
    def root_dir(self):
        """Returns the root directory for the session.

        Returns:
            str: The root directory for the session.
        """

        return os.path.join(BASE_DIR, self.session_id)


class PostgresAgentInstruments(AgentInstruments):
    """
    Unified Toolset for the postgres Data Analytics Multi-Agent System

    This class provides a standardized set of tools and functions for agents within a multi-agent system 
    that interacts with a Postgres database. It offers several advantages:

    - **Shared State and Functions:** All agents have access to the same state (like database connection) 
      and functions, promoting consistency and reducing redundancy.
    - **Context Awareness:** Agents can utilize functions that dynamically adapt to changing context 
      based on the current session.
    - **Clear Agent Capabilities:** Functions offer well-defined capabilities for agents, 
      facilitating easier development and understanding.
    - **Clean Database Management:** The class simplifies database connection management.

    **Guidelines:**

    - **Agent Function Isolation:** Agent functions should not directly call other agent functions. 
      Instead, they should interact with external lower-level modules.
    - **One-to-One Mapping (Optional):** Consider a one-to-one mapping between agents and their functions 
      for simpler architecture.
    - **Persistent State Lifecycle:** The state of this class persists across agent orchestrations.
    """

    def __init__(self, db_url: str, session_id: str) -> None:
        """Initializes the class instance.

        Args:
            db_url (str): The URL for connecting to the Postgres database.
            session_id (str): The unique identifier for the current session.
        """

        super().__init__()

        self.db_url = db_url
        self.db = None
        self.session_id = session_id
        self.messages = []
        self.complete_keyword = "APPROVED"

        self.innovation_index = 0

    def __enter__(self):
        """Context manager entry point.

        Resets files in the root directory and establishes a connection to the Postgres database.

        Returns:
            tuple: A tuple containing itself and the connected PostgresManager object.
        """

        self.reset_files()
        self.db = PostgresManager()
        self.db.connect_with_url(self.db_url)
        return self, self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point.

        Closes the database cursor and connection if they are open.
        """
        if self.db.cur:
            self.db.cur.close()
        if self.db.conn:
            self.db.conn.close()

    def sync_messages(self, messages: list):
        """Synchronizes messages with the orchestrator.

        Args:
            messages (list): A list of messages to synchronize.
        """
        self.messages = messages

    def reset_files(self):
        """Clears all files within the root directory.

        Creates the directory if it doesn't exist.
        """

        # create it if it doesn't exist
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        for fname in os.listdir(self.root_dir):
            os.remove(os.path.join(self.root_dir, fname))

    def get_file_path(self, fname: str):
        """Generates the absolute path for a file within the root directory.

        Args:
            fname (str): The filename.

        Returns:
            str: The absolute path to the file.
        """

        return os.path.join(self.root_dir, fname)

    # ------------------------------ Agent Properties ------------------------------

    @property
    def run_sql_results_file(self):
        """
        Returns the path to the file containing the results of the last run_sql call.
        """

        return self.get_file_path("run_sql_results.json")

    # ------------------------------ Agent Functions ------------------------------

    def run_sql(self, sql: str) -> str:
        """Executes a provided SQL query against the Postgres database.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            str: A message indicating successful execution or an error message.

        Raises:
            Exception: If an error occurs during database interaction.
        """

        results_as_json = self.db.run_sql(sql)

        fname = self.run_sql_results_file

        with open(fname, "w") as f:
            f.write(results_as_json)

        return "Successfully delivered results to json file"

    def validate_run_sql(self):
        """Checks if the "run_sql_results.json" file exists and contains content.

        This method validates the results of the `run_sql` function. It verifies 
        whether the file containing the query results exists in the agent's root 
        directory and if the file has any content.  

        Returns:
            bool: True if the file exists and has content, False otherwise.
        """

        fname = self.run_sql_results_file

        with open(fname, "r") as f:
            content = f.read()

        if content:
            return True, ""
        else:
            return False, f"File {fname} is empty"

    def write_file(self, content: str):
        """
        Writes content to a file named "write_file.txt" in the agent's root directory.

        This method is a generic file writer that takes a string as content and writes it
        to a file named "write_file.txt" within the agent's root directory. 

        Args:
            content (str): The content to be written to the file.
        """

        fname = self.get_file_path("write_file.txt")
        return file.write_file(fname, content)

    def write_json_file(self, content: str):
        """
        Writes content to a file named "write_json_file.json" in the agent's root directory.

        This method is similar to `write_file` but specifically writes JSON content 
        to a file named "write_json_file.json".

        Args:
            content (str): The JSON content to be written to the file.
        """

        fname = self.get_file_path("write_json_file.json")
        return file.write_json_file(fname, content)

    def write_yml_file(self, content: str):
        """
        Writes content to a file named "write_yml_file.yml" in the agent's root directory.

        This method is similar to `write_file` but specifically writes YAML content 
        to a file named "write_yml_file.yml".

        Args:
            content (str): The YAML content to be written to the file.
        """

        fname = self.get_file_path("write_yml_file.yml")
        return file.write_yml_file(fname, content)

    def write_innovation_file(self, content: str):
        """
        Writes content to a numbered innovation file in the agent's root directory.

        This method creates a new file named with a sequential index 
        (e.g., "0_innovation_file.txt", "1_innovation_file.txt") and writes the 
        provided content to the file. The index increments for each subsequent call.

        Args:
            content (str): The content to be written to the file.

        Returns:
            str: A success message indicating the file was written.
        """

        fname = self.get_file_path(f"{self.innovation_index}_innovation_file.json")
        file.write_file(fname, content)
        self.innovation_index += 1
        return f"Successfully wrote innovation file."

    def validate_innovation_files(self):
        """
        Validates the existence and content of innovation files.

        This method checks if all innovation files within the specified range exist and contain content.
        It iterates through the files, reading their contents and verifying that they are not empty.

        Returns:
            bool: True if all innovation files are valid, False otherwise.
        """

        # loop from 0 to innovation_index and verify file exists with content
        for i in range(self.innovation_index):
            fname = self.get_file_path(f"{i}_innovation_file.json")
            with open(fname, "r") as f:
                content = f.read()
                if not content:
                    return False, f"File {fname} is empty"

        return True, ""