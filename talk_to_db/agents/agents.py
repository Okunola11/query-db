
import autogen
from typing import Optional

from talk_to_db.modules.db import PostgresManager
from talk_to_db.agents import agent_config
from talk_to_db.modules import orchestrator
from talk_to_db.agents.instruments import PostgresAgentInstruments

#  ------------------- PROMPTS -------------------

# create our terminate message
def is_termination_msg(content):
    """Checks if the given content indicates a termination message.

    Used by AI agents to determine when to terminate a conversation or process.

    Args:
        content (dict): A dictionary containing the message content.

    Returns:
        bool: True if the content indicates a termination message, False otherwise.

    Notes:
        A termination message is defined as a message containing the string "APPROVED".
    """

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

TEXT_REPORT_ANALYST_PROMPT = "Text File Report Analyst. You exclusively use the write_file function on a summarized report."

JSON_REPORT_ANALYST_PROMPT = "JSON Report Analyst. You exclusively use the write_json_file function on the report."

YML_REPORT_ANALYST_PROMPT = "YAML Report Analyst. You exclusively use the write_yml_file function on the report."

# ------------------- AGENTS -------------------

def build_data_engr_team(instruments: PostgresAgentInstruments):
    """Builds a team of agents to manage data engineering tasks.

    This function creates a team of agents, including a user proxy agent, 
    a data engineer agent, a senior data analyst agent, and a product manager agent.
    
    Args:
        instruments (PostgresAgentInstruments): The instruments to be used by the agents.

    Returns:
        list: A list of agents in the data engineering team.
    """

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
        llm_config=agent_config.base_config,
        system_message=DATA_ENGINEER_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg
    )

    # sr data analyst agent - runs the sql query and generates the response
    sr_data_analyst = autogen.AssistantAgent(
        name="Sr_Data_Analyst",
        llm_config=agent_config.run_sql_config,
        system_message=SR_DATA_ANALYST_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        function_map={"run_sql": instruments.run_sql}
    )

    # product manager - validates the response to make sure it's correct
    product_manager = autogen.AssistantAgent(
        name="Product_Manager",
        llm_config=agent_config.base_config,
        system_message=PRODUCT_MANAGER_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg
    )

    return [user_proxy, data_engineer, sr_data_analyst]


def build_data_report_team(instruments: PostgresAgentInstruments):
    """Builds a team of agents to manage data reporting tasks.

    This function creates a team of agents, including a user proxy agent, 
    a text report analyst agent, a JSON report analyst agent, and a YAML report analyst agent.
    
    Args:
        instruments (PostgresAgentInstruments): The instruments to be used by the agents.

    Returns:
        list: A list of agents in the data reporting team.
    """

    # Admin user proxy agent - takes in the prompt and manages the group chat
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMT,
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg
    )

    # text report analyst - writes a summary report of the result and saves them to a local text file
    text_report_analyst = autogen.AssistantAgent(
        name="Text_Report_Analyst",
        llm_config=agent_config.write_file_config,
        system_message=TEXT_REPORT_ANALYST_PROMPT,
        function_map={"write_file": instruments.write_file}
    )

    # json_report analyst - writes a summary report of the results and saves them to a local json file
    json_report_analyst = autogen.AssistantAgent(
        name="JSON_Report_Analyst",
        llm_config=agent_config.write_json_file_config,
        system_message=JSON_REPORT_ANALYST_PROMPT,
        function_map={"write_json_file": instruments.write_json_file}
    )

    # yaml report analyst - writes a summary report of the results and saves them to a local yaml file
    yml_report_analyst = autogen.AssistantAgent(
        name="YML_Report_Analyst",
        llm_config=agent_config.write_yml_file_config,
        system_message=YML_REPORT_ANALYST_PROMPT,
        function_map={"write_yml_file": instruments.write_yml_file}
    )

    return [user_proxy, text_report_analyst, json_report_analyst, yml_report_analyst]

# ------------------- ORCHESTRATION -------------------

def build_team_orchestrator(
    team: str,
    agent_instruments: PostgresAgentInstruments,
    validate_results: callable = None
) -> orchestrator.Orchestrator:
    """Builds an orchestrator for the specified team.

    Args:
        team (str): The team name. Currently supported teams are "data_engr" and "data_report".
        agent_instruments (PostgresAgentInstruments): The instruments to be used by the agents.
        validate_results (callable, optional): A function to validate the results. Defaults to None.

    Returns:
        orchestrator.Orchestrator: The built orchestrator instance for the specified team.

    Notes:
        The orchestrator is configured with a set of agents specific to the chosen team.
        If the team is not recognized, an Exception is raised.
    """

    if team == "data_engr":
        return orchestrator.Orchestrator(
            name="Postgres Data Analytics Multi-Agent ::: Data Engineering Team",
            agents=build_data_engr_team(agent_instruments),
            instruments=agent_instruments,
            validate_results_func=validate_results
        )
    elif team == "data_report":
        return orchestrator.Orchestrator(
            name="Postgres Data Analytics Multi-Agent ::: Data Report Team",
            agents=build_data_report_team(agent_instruments),
            instruments=agent_instruments,
            validate_results_func=validate_results
        )

    raise Exception("Unknown team: " + team)