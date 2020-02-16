FROM continuumio/miniconda3:latest

WORKDIR /app
COPY ./data_engineering/archive_etler/environment.yml /app/environment.yml
RUN conda env create -f ./environment.yml

RUN echo "source activate app-env" > ~/.bashrc
ENV PATH /opt/conda/envs/app-env/bin:$PATH

COPY data_engineering /app/data_engineering
COPY splits /app/splits
COPY ./.gcs /app/.gcs

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"
# ENV GOOGLE_APPLICATION_CREDENTIALS /app/.gcs/credentials.json

CMD ["python", "./data_engineering/archive_etler/main.py"]