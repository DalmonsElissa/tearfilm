FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/tearfilm/src

WORKDIR /opt/tearfilm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e . \
    && python -c "import tearfilm_persistence; print(tearfilm_persistence.__version__)"

ENTRYPOINT ["python", "-m"]
CMD ["tearfilm_persistence.cli.verify"]
