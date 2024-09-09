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

        (
            assistant.get_or_create_assistant(assistant_name)
            .set_instructions(
                "You are an elite SQL developer. You generate the most concise and performant SQL queries"
            )
            .equip_tools(tools)
            .make_thread()
            .add_message(prompt)
            .run_thread()
            .add_message(
                "Use the run_sql function to run the SQL you have just generated."
            )
            .run_thread(toolbox=[tools[0].name])
            .run_validation(agent_instruments.validate_run_sql)
            .spy_on_assistant(agent_instruments.make_agent_chat_file(assistant_name))
            .get_cost_and_tokens(agent_instruments.make_agent_cost_file(assistant_name))
        )

        print(f"✅ Turbo4 Assistant finished.")


if __name__ == "__main__":
    main()