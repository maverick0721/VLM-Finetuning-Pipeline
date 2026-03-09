import pypandoc
import os


INPUT_FILE = "reports/experiment_report.md"
OUTPUT_FILE = "reports/experiment_report.pdf"


def main():

    if not os.path.exists(INPUT_FILE):

        print("Markdown report not found.")
        return

    pypandoc.convert_file(
        INPUT_FILE,
        "pdf",
        outputfile=OUTPUT_FILE
    )

    print("PDF report generated:", OUTPUT_FILE)


if __name__ == "__main__":
    main()