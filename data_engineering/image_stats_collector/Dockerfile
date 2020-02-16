FROM python:3.6-alpine as base
FROM base as builder

ARG filedir

#RUN apk --no-cache add bash curl
#RUN curl -sSO https://dl.google.com/cloudagents/install-monitoring-agent.sh
#RUN bash install-monitoring-agent.sh

RUN echo "filedir" $filedir

RUN mkdir /install
WORKDIR /install

COPY data_engineering/requirements.txt /requirements.txt
RUN pip install --install-option="--prefix=/install" -r /requirements.txt

FROM base
COPY --from=builder /install /usr/local
COPY ./data_engineering /app/data_engineering
COPY ./.gcs /app/.gcs

WORKDIR /app

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "/app/data_engineering/image_stats_collector/main.py"]