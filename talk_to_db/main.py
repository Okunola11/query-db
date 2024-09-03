import argparse
import autogen

from talk_to_db.modules.db import PostgresManager
from talk_to_db.modules import llm, orchestrator, file
from talk_to_db.settings import DB_URL
from talk_to_db.settings import OPENAI_API_KEY

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

        # build the gpt configuration object
        # base configuration
        base_config = {
            "temperature": 0,
            "config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}],
            "timeout": 120,
        }

        # configuration with 'run_sql'
        run_sql_config = {
            **base_config,
            "functions": [
                {
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
                },
            ]
        }

        # configuration with 'write_file'
        write_file_config = {
            **base_config,
            "functions": [
                {
                    "name": "write_file",
                    "description": "Write a file to the filesystem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fname": {
                                "type": "string",
                                "description": "The name of the file to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the file to write"
                            }
                        },
                        "required": ['fname', 'content'],
                    },
                },
            ]
        }

        # configuration with 'write_json_file'
        write_json_file_config = {
            **base_config,
            "functions": [
                {
                    "name": "write_json_file",
                    "description": "Write a json file to the filesystem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fname": {
                                "type": "string",
                                "description": "The name of the file to write",
                            },
                            "json_str": {
                                "type": "string",
                                "description": "The content of the file to write"
                            }
                        },
                        "required": ['fname', 'json_str']
                    }
                }
            ]
        }

        # configuration with 'write_yaml_file'
        write_yml_file_config = {
            **base_config,
            "functions": [
                {
                    "name": "write_yml_file",
                    "description": "Write a yaml file to the filesystem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fname": {
                                "type": "string",
                                "description": "The name of the file to write",
                            },
                            "json_str": {
                                "type": "string",
                                "description": "The content of the file to write"
                            }
                        },
                        "required": ['fname', 'json_str'],
                    },
                },
            ]
        }

        # build the function map
        function_map_run_sql = {
            "run_sql": db.run_sql,
        }

        function_map_write_file = {
            "write_file": file.write_file,
        }

        function_map_write_json_file = {
            "write_json_file": file.write_json_file,
        }

        function_map_write_yml_file = {
            "write_yml_file": file.write_yml_file,
        }

        # create our terminate message
        def is_termination_msg(content):
            have_content = content.get("content", None)
            if have_content and "APPROVED" in content["content"]:
                return True
            return False

        COMPLETION_PROMPT = "If everything looks good, respond with APPROVED"

        USER_PROXY_PROMT = "A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin."

        DATA_ENGINEER_PROMPT = "A Data Engineer. You follow an approved plan. Generate the initial SQL based on the requirements provided. Send it to the Sr Data Analyst to be executed."

        SR_DATA_ANALYST_PROMPT = "Sr Data Analyst. You follow an approved plan. You run the SQL query, generate the response and send it to the product manager for final review"

        PRODUCT_MANAGER_PROMPT = (
            "Product Manager. Validate the response to make sure it is correct."
            + COMPLETION_PROMPT
        )

        # Creating agents with specific roles

        # Admin user proxy agent - takes in the prompt and manages the group chat
        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message=USER_PROXY_PROMT,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg
        )

        # data engineer agent - generated the sql query
        data_engineer = autogen.AssistantAgent(
            name="Engineer",
            llm_config=base_config,
            system_message=DATA_ENGINEER_PROMPT,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg
        )

        # sr data analyst agent - runs the sql query and generates the response
        sr_data_analyst = autogen.AssistantAgent(
            name="Sr_Data_Analyst",
            llm_config=run_sql_config,
            system_message=SR_DATA_ANALYST_PROMPT,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
            function_map=function_map_run_sql
        )

        # product manager - validates the response to make sure it's correct
        product_manager = autogen.AssistantAgent(
            name="Product_Manager",
            llm_config=base_config,
            system_message=PRODUCT_MANAGER_PROMPT,
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg
        )

        data_engineering_agents = [user_proxy, data_engineer, sr_data_analyst, product_manager]

        data_engr_orchestrator = orchestrator.Orchestrator(
            name="Postgres Data Analytics Multi-Agent ::: Data Engineering Team",
            agents=data_engineering_agents
        )

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