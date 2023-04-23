import logging

CONFIG_FILE = 'requirements.txt'
BACKUP_SUFFIX = '.backup'


def read_config_file():
    logging.info(f"Reading {CONFIG_FILE}...")
    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()
        return lines


def create_backup_file(contents):
    backup_file_name = CONFIG_FILE + BACKUP_SUFFIX
    logging.info(f"Creating {backup_file_name}...")
    with open(backup_file_name, 'w') as f:
        f.writelines(contents)


def move_to_latest(libraries):
    logging.debug("Moving library versions to 'latest available'...")
    updated_libraries = remove_version_suffix(libraries)
    overwrite_config_file(updated_libraries)


def remove_version_suffix(libraries):
    logging.debug("Removing version numbers from each line...")
    # Need to add the leading newline
    return [library.split("==")[0] + '\n' for library in libraries]


def overwrite_config_file(contents):
    logging.info(f"Overwriting {CONFIG_FILE}...")
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(contents)


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logging.debug(f"Logging level set to: {logging._levelToName[level]} ({level})")


def main():
    setup_logging()
    original_contents = read_config_file()
    create_backup_file(original_contents)
    move_to_latest(original_contents)


if __name__ == "__main__":
    main()
