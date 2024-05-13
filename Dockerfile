FROM python:3.10

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry
RUN poetry install --no-dev

COPY . .

CMD ["python", "-m", "unittest"]
