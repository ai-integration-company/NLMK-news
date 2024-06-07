FROM huecker.io/library/python:3.9-slim

WORKDIR /ml

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download ru_core_news_sm

EXPOSE 3000


CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]
