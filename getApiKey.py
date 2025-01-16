def get_key():
    with open("apiKey", "r") as key_file:
        return key_file.read()