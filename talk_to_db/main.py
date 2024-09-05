import argparse

from talk_to_db.modules.db import PostgresManager
from talk_to_db.modules import llm, embeddings, rand
from talk_to_db.settings import DB_URL
from talk_to_db.agents import agents
from talk_to_db.agents.instruments import PostgresAgentInstruments
from talk_to_db.types import ConversationResult


POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"


def main():
    """
    The main function for the Postgres AI agent.

    This function performs the following steps:

    1. Parses command-line arguments for the prompt.
    2. Validates the presence of the prompt and exits if missing.
    3. Creates a user-friendly prompt by prepending "Fulfill this database query:".
    4. Generates a session ID for tracking.
    5. Establishes a connection to the Postgres database and retrieves table definitions.
    6. Creates a `DatabaseEmbedder` instance to embed table information.
    7. Adds retrieved table definitions to the `DatabaseEmbedder`.
    8. Identifies similar tables based on the user prompt using embeddings.
    9. Extracts definitions for the identified similar tables.
    10. Enhances the prompt by adding a reference to the retrieved table definitions.
    11. Constructs a `DataEngineeringOrchestrator` object with validation capabilities.
    12. Orchestrates a sequential conversation with the data engineering team using the enhanced prompt.
    13. Gets the conversation results, including success status, cost, and tokens.

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    raw_prompt = args.prompt

    prompt = f"Fulfill this database query: {raw_prompt}"

    session_id = rand.generate_session_id(raw_prompt)

    with PostgresAgentInstruments(DB_URL, session_id) as (agent_instruments, db):
        # ---------------- GATE TEAM: Prevents bad prompts from running ----------------

        gate_orchestrator = agents.build_team_orchestrator(
            "scrum_master",
            agent_instruments,
            validate_results=lambda: (True, "")
        )

        gate_orchestrator_result: ConversationResult = (
            gate_orchestrator.sequential_conversation(prompt)
        )

        print("gate_orchestrator.last_message_str", gate_orchestrator_result.last_message_str)

        nlq_confidence = int(gate_orchestrator_result.last_message_str)

        match nlq_confidence:
            case (1 | 2):
                print(f"❌ Gate Team Rejected - Confidence too low: {nlq_confidence}")
                return
            case (3 | 4 | 5):
                print(f"✅ Gate Team Approved - Valid confidence: {nlq_confidence}")
                print(f"💰📊🤖 Gate team Cost: {gate_orchestrator_result.cost}, tokens: {gate_orchestrator_result.tokens}")
            case _:
                print("❌ Gate Team Rejected - Invalid response")
                return

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

        # -------------------------------- DATA ENGINEERING TEAM --------------------------------

        data_engr_orchestrator = agents.build_team_orchestrator(
            "data_engr",
            agent_instruments,
            validate_results=agent_instruments.validate_run_sql
        )

        data_engr_conversation_result: ConversationResult = (
            data_engr_orchestrator.sequential_conversation(prompt)
        )

        match data_engr_conversation_result:
            case ConversationResult(
                success=True, cost=data_engr_cost, tokens=data_engr_tokens
            ):
                print(f"✅ Orchestrator was successful. Team: {data_engr_orchestrator.name}")

                print(f"Data Engr Cost: {data_engr_cost}, tokens: {data_engr_tokens}")

                print(f"💰📊🤖 Organization Cost: {data_engr_cost}, tokens: {data_engr_tokens}")
            case _:
                print(f"❌ Orchestrator failed. Team: {data_engr_orchestrator.name} Failed.")


if __name__ == "__main__":
    main()