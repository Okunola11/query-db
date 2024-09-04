from talk_to_db.modules import file
from talk_to_db.modules.db import PostgresManager
from talk_to_db.settings import OPENAI_API_KEY


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

# ------------------- FUNCTION MAPS -------------------

def create_function_map(name: str, func: callable):
    """Creates a dictionary mapping a function name to its implementation.

    Args:
        name (str): The name of the function.
        func (callable): The function implementation.

    Returns:
        dict: A dictionary with a single entry, where the key is the function name and the value is the function implementation.
    """

    return {name: func}

def build_function_map_run_sql(db: PostgresManager):
    """Builds a function map for running SQL queries using the provided PostgresManager instance.

    Args:
        db (PostgresManager): The PostgresManager instance for database interactions.

    Returns:
        dict: A dictionary with a single entry, where the key is "run_sql" and the value is the db.run_sql method.
    """

    return create_function_map("run_sql", db.run_sql)

function_map_write_file = create_function_map("write_file", file.write_file) 
    
function_map_write_json_file = create_function_map("write_json_file", file.write_json_file)

function_map_write_yml_file = create_function_map("write_yml_file", file.write_yml_file)