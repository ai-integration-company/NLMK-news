FROM python:3.9

WORKDIR /scrapper

COPY requirements.txt ./
COPY news_sources.json ./
ENV POETRY_VERSION=1.1.12

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["sh", "entrypoint.sh"]
