def get_key(file_name):
    with open(file_name, "r") as key_file:
        return key_file.read()