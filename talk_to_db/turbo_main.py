import argparse
from typing import List, Callable

from talk_to_db.agents.turbo4 import Turbo4
from talk_to_db.types import Chat, TurboTool
from talk_to_db.agents.instruments import PostgresAgentInstruments
from talk_to_db.modules import llm, rand, embeddings
from talk_to_db.settings import DB_URL


POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"


run_sql_tool_config = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Run a SQL query against the postgres database",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to run",
                }
            },
            "required": ['sql'],
        },
    }
}



def main():
    """
    Main function for executing a database query using the Turbo4 assistant and PostgresAgentInstruments.

    This function performs the following steps:
    1. Parses command-line arguments to get the AI prompt.
    2. Validates the presence of a prompt and constructs a database query prompt.
    3. Initializes a Turbo4 assistant and a unique session ID for tracking.
    4. Sets up a connection to a PostgreSQL database using `PostgresAgentInstruments`.
    5. Retrieves and embeds table definitions from the database.
    6. Identifies similar tables based on the provided prompt and updates the prompt with these table definitions.
    7. Configures the Turbo4 assistant with a specific set of tools and instructions.
    8. Creates a thread for interacting with the assistant and processes the prompt to generate SQL queries.
    9. Runs the generated SQL queries and validates their execution.
    10. Monitors the assistant's activities and calculates associated costs and tokens.
    11. Outputs a confirmation message once the process is complete.

    Command-line Arguments:
        --prompt (str): The prompt for the AI, specifying the database query to fulfill.

    Process Flow:
        - Parse the prompt argument.
        - Initialize database embeddings and identify similar tables.
        - Update the prompt with table definitions.
        - Create and configure the Turbo4 assistant with tools.
        - Execute and validate SQL queries.
        - Record and report the assistant's activities and costs.

    Raises:
        ValueError: If the prompt is not provided.
        Other exceptions: Errors related to database operations or assistant interactions.

    Example:
        $ poetry run turbo --prompt "Which users with pro subscription are located in LA"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    raw_prompt = args.prompt

    prompt = f"Fulfill this database query: {raw_prompt}"

    assistant_name = "Turbo4"

    assistant = Turbo4()

    session_id = rand.generate_session_id(assistant_name + raw_prompt)

    with PostgresAgentInstruments(DB_URL, session_id) as (agent_instruments, db):

        # ---------------- BUILDING TABLE DEFINITIONS ----------------

        map_table_name_to_table_def = db.get_table_definition_map_for_embeddings()

        database_embedder = embeddings.DatabaseEmbedder()

        for name, table_def in map_table_name_to_table_def.items():
            database_embedder.add_table(name, table_def)

        similar_tables = database_embedder.get_similar_tables(raw_prompt, n=2)
        print("\n---------------- SIMILAR TABLES ---------------")
        print(similar_tables)

        table_definitions = database_embedder.get_table_definitions_from_names(
            similar_tables
        )
    
        prompt = llm.add_cap_ref(
            prompt,
            f"Use these {POSTGRES_TABLE_DEFINITIONS_CAP_REF} to satisfy the database query.",
            POSTGRES_TABLE_DEFINITIONS_CAP_REF,
            table_definitions
        )

        # -------------------------------- ASSISTANT --------------------------------

        tools = [
            TurboTool("run_sql", run_sql_tool_config, agent_instruments.run_sql)
        ]

        # (
        #     assistant.get_or_create_assistant(assistant_name)
        #     .set_instructions(
        #         "You are an elite SQL developer. You generate the most concise and performant SQL queries"
        #     )
        #     .equip_tools(tools)
        #     .make_thread()
        #     .add_message(prompt)
        #     .run_thread()
        #     .add_message(
        #         "Use the run_sql function to run the SQL you have just generated."
        #     )
        #     .run_thread(toolbox=[tools[0].name])
        #     .run_validation(agent_instruments.validate_run_sql)
        #     .spy_on_assistant(agent_instruments.make_agent_chat_file(assistant_name))
        #     .get_cost_and_tokens(agent_instruments.make_agent_cost_file(assistant_name))
        # )

        # print(f"✅ Turbo4 Assistant finished.")

    # -------------------- SIMPLE PROMPT SOLUTION - 2 API CALLS

        sql_response = llm.prompt(
            prompt,
            model="gpt-4o-mini",
            instructions="You are an elite SQL developer. You generate the most concise and performant SQL queries."
        )

        response = llm.prompt_func(
            "Use the run_sql function to run the SQL you have just generated: "
            + sql_response,
            model="gpt-4o-mini",
            instructions="You are an elite SQL developer. You generate the most concise and performant SQL queries.",
            turbo_tools=tools
        )
        print(f"\n\nprompt_funct() respnse is: {response}")

        agent_instruments.validate_run_sql()


if __name__ == "__main__":
    main()