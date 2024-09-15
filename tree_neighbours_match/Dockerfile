FROM python:3.12-bookworm

RUN pip install poetry==1.8.3

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY train_agent.py ./train_agent.py
COPY dataset.py ./dataset.py
COPY models.py ./models.py


RUN poetry install 

ENTRYPOINT ["poetry", "run", "python", "train_agent.py"]
