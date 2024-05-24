import yaml


def load_params():
    with open("params.yaml") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise "Could not load params.yaml"
    return params
