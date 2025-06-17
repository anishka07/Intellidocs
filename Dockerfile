FROM python:3.10

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv
RUN uv sync

COPY app .

CMD [ "uv", "run", "fastapi", "run" ]