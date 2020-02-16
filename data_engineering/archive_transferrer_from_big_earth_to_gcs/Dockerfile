FROM python:3.6 as base
FROM base as builder

ARG filedir

RUN echo "filedir" $filedir

RUN mkdir /install
WORKDIR /install

COPY data_engineering/archive_transferrer_from_big_earth_to_gcs/requirements.txt /requirements.txt
RUN pip install --install-option="--prefix=/install" -r /requirements.txt

FROM base
COPY --from=builder /install /usr/local
COPY ./data_engineering /app/data_engineering
COPY ./.gcs /app/.gcs

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "./data_engineering/archive_transferrer_from_big_earth_to_gcs/main.py"]