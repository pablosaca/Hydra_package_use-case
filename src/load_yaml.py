import yaml


with open("../config/config_file.yaml", "r") as file:
    print(yaml.safe_load(file))
