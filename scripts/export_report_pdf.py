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
        outputfile=OUTPUT_FILE,
        extra_args=[
            "--pdf-engine=wkhtmltopdf",
            "--metadata",
            "title=VLM Finetuning Experiment",
            "--pdf-engine-opt=--enable-local-file-access"
        ]
    )

    print("PDF report generated:", OUTPUT_FILE)


if __name__ == "__main__":
    main()