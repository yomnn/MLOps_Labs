FROM python:3.12-slim-bookworm

ENV PORT=8000

WORKDIR /app

COPY req.txt /app/req.txt

RUN pip install -r /app/req.txt

COPY . /app

EXPOSE ${PORT}

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
