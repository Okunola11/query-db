import json
import yaml

def write_file(fname, content):
    with open(fname, "w") as f:
        f.write(content)

def write_json_file(fname, json_str: str):
    # replace any ' with "
    json_str = json_str.replace("'", '"')

    # safely convert the JSON string to a python object
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # write the Python Object to the file as JSON
    with open(fname, "w") as f:
        json.dump(data, f, indent=4)

def write_yml_file(fname, json_str: str):
    # replace ' with "
    json_str = json_str.replace("'", '"')

    try:
        data = yaml.safe_load(json_str)
    except yaml.error as e:
        print(f"Error decoding JSON: {e}")
        return

    # write the python object to the file as YAML
    with open(fname, "w") as f:
        yaml.dump(data, f)
