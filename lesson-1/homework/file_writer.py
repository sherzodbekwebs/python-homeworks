from main import create_virtual_env, install_packages


class FileWriter:
    @staticmethod
    def write_file(file_path, content):
        with open(file_path, "w") as file:
            file.write(content)

if __name__ == "__main__":
    create_virtual_env()
    install_packages()
