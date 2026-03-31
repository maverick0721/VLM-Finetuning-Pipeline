import os
import shutil

import pypandoc


INPUT_FILE = "reports/experiment_report.md"
OUTPUT_FILE = "reports/experiment_report.pdf"


def pick_pdf_engine():
    for engine in ["wkhtmltopdf", "xelatex", "pdflatex"]:
        if shutil.which(engine):
            return engine
    return None


def main():
    if not os.path.exists(INPUT_FILE):
        print("Markdown report not found.")
        return

    engine = pick_pdf_engine()
    if engine is None:
        print("No supported PDF engine found (wkhtmltopdf/xelatex/pdflatex). Skipping PDF export.")
        return

    extra_args = ["--pdf-engine", engine, "--metadata", "title=VLM Finetuning Experiment"]
    if engine == "wkhtmltopdf":
        extra_args.append("--pdf-engine-opt=--enable-local-file-access")

    pypandoc.convert_file(INPUT_FILE, "pdf", outputfile=OUTPUT_FILE, extra_args=extra_args)
    print("PDF report generated:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
