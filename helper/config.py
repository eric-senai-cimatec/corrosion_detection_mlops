import yaml


def load_config(*keys):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Se nenhuma chave for passada, retorna o dicionário inteiro
    if not keys:
        return config

    # Navega pelo dicionário caso haja chaves aninhadas (ex: config['data']['path'])
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None

    return value
