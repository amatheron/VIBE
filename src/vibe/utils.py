"""
Useful utilities.
"""

import yaml

def read_yaml(yaml_file):
    """
    Read a YAML file and return its contents.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.

    Returns
    -------
    dict
        Contents of the YAML file.
    """
    with open(yaml_file) as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    """
    Write data to a YAML file.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.
    data : dict
        Data to write to the YAML file.
    """
    with open(yaml_file, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)