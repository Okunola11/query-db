import argparse

from talk_to_db.modules.db import PostgresManager
from talk_to_db.modules import llm
from talk_to_db.settings import DB_URL

POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"
RESPONSE_FORMAT_CAP_REF = "RESPONSE_FORMAT"

SQL_DELIMITER = "----------"

def main():
    """The main function for the Postgres AI agent.

    This function parses command-line arguments, generates a prompt for the AI, and executes
    the generated SQL query using the PostgresManager.

    Args:
        None

    Returns:
        None
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    prompt = f"Fulfill this database query: {args.prompt}"

    with PostgresManager() as db:
        db.connect_with_url(DB_URL)

        table_definitions = db.get_table_definitions_for_prompt()

        prompt = llm.add_cap_ref(
            prompt,
            f"Use these {POSTGRES_TABLE_DEFINITIONS_CAP_REF} to satisfy the database query.",
            POSTGRES_TABLE_DEFINITIONS_CAP_REF,
            table_definitions
        )

        prompt = llm.add_cap_ref(
            prompt,
            f"\n\nRespond in this format {RESPONSE_FORMAT_CAP_REF}. Replace the text between <> with it's request. I need to be able easily parse the sql query from your response.",
            RESPONSE_FORMAT_CAP_REF,
            f"""<explanation of the sql query>
            {SQL_DELIMITER}
            <sql query exclusively as raw text>"""
        )

        print("\n\n--------- PROMPT ----------")
        print(prompt)

        prompt_response = llm.prompt(prompt)

        print("\n\n--------- PROMPT RESPONSE ---------")
        print(prompt_response)

        sql_query = prompt_response.split(SQL_DELIMITER)[1].strip()

        print(f"\n\n--------- PARSED SQL QUERY ---------")
        print(sql_query)

        result = db.run_sql(sql_query)

        print("\n\n========== POSTGRES DATA ANALYTICS AI AGENT RESPONSE ==========")
        print(result)



if __name__ == "__main__":
    main()