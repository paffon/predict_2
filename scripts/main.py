# Main module

from classes.project_handler import ProjectHandler


def main():
    project_handler = ProjectHandler()

    # project_handler.get_data()
    project_handler.extract_features()
    project_handler.train()
    project_handler.report_training_outcome()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
