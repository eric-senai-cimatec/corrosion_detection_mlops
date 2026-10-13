import yaml


def load_config(path):
    """Loads settings from data\data.yaml and allows retrieving specific or nested keys."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
