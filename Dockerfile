FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -e .

COPY ardrag ./ardrag
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

RUN mkdir -p /data/uploads

EXPOSE 8000 8001

ENTRYPOINT ["./entrypoint.sh"]
