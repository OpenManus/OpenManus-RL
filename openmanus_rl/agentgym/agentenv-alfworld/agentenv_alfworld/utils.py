import yaml


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def load_config(config_file):
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config
