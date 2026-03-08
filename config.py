import yaml, re

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        path = re.sub(r'\\', '/', f.read())
        return yaml.safe_load(path)