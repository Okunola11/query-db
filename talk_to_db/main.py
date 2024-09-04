import argparse
import autogen

from talk_to_db.modules.db import PostgresManager
from talk_to_db.modules import llm, orchestrator, file, embeddings
from talk_to_db.settings import DB_URL
from talk_to_db.settings import OPENAI_API_KEY
from talk_to_db.agents import agents

POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"
RESPONSE_FORMAT_CAP_REF = "RESPONSE_FORMAT"

SQL_DELIMITER = "----------"

def main():
    """
    The main function for the Postgres AI agent.

    This function parses command-line arguments, generates a prompt for the AI, and executes
    the generated SQL query using the PostgresManager.

    It now includes the following steps:
        1. Retrieves table definitions from the database.
        2. Creates a `DatabaseEmbedder` instance and adds table definitions to it.
        3. Finds similar tables based on the prompt.
        4. Adds table definitions to the prompt as a reference.
        5. Orchestrates a conversation with a team of data engineering agents.
        6. Prints the results and cost information.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    raw_prompt = args.prompt

    prompt = f"Fulfill this database query: {raw_prompt}"

    with PostgresManager() as db:
        db.connect_with_url(DB_URL)

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

        data_engr_orchestrator = agents.build_team_orchestrator("data_engr", db)

        success, data_engr_messages = data_engr_orchestrator.sequential_conversation(prompt)

        print(f"---------- DATA ENGR RESULTS ----------")
        print(f"Data Engineer Result is: {data_engr_messages[-2]["content"]}")
        data_engr_results = data_engr_messages[-2]['content']


        # -------------------------------- REPORT ANALYST AGENTS -----------------------------------------------------

        TEXT_REPORT_ANALYST_PROMPT = "Text File Report Analyst. You exclusively use the write_file function on a summarized report."
        JSON_REPORT_ANALYST_PROMPT = "JSON Report Analyst. You exclusively use the write_json_file function on the report."
        YML_REPORT_ANALYST_PROMPT = "YAML Report Analyst. You exclusively use the write_yml_file function on the report."

        # text report analyst - writes a summary report of the result and saves them to a local text file
        text_report_analyst = autogen.AssistantAgent(
            name="Text_Report_Analyst",
            llm_config=write_file_config,
            system_message=TEXT_REPORT_ANALYST_PROMPT,
            function_map=function_map_write_file
        )

        # json_report analyst - writes a summary report of the results and saves them to a local json file
        json_report_analyst = autogen.AssistantAgent(
            name="JSON_Report_Analyst",
            llm_config=write_json_file_config,
            system_message=JSON_REPORT_ANALYST_PROMPT,
            function_map=function_map_write_json_file
        )

        # yaml report analyst - writes a summary report of the results and saves them to a local yaml file
        yml_report_analyst = autogen.AssistantAgent(
            name="YML_Report_Analyst",
            llm_config=write_yml_file_config,
            system_message=YML_REPORT_ANALYST_PROMPT,
            function_map=function_map_write_yml_file
        )

        data_report_agents = [user_proxy, text_report_analyst, json_report_analyst, yml_report_analyst]

        data_report_orchestrator = orchestrator.Orchestrator(
            name="Postgres Data Analytics Multi-Agent ::: Data Report Team",
            agents=data_report_agents
        )

        data_report_prompt = f"Here is the data to report: {data_engr_results}"

        if data_engr_results:
            data_report_orchestrator.broadcast_conversation(data_report_prompt)


if __name__ == "__main__":
    main()