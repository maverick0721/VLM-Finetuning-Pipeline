FROM python:3.10

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y git graphviz pandoc wkhtmltopdf && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "run_pipeline.py"]
