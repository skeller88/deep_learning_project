# https://hub.docker.com/r/continuumio/miniconda3/tags
FROM continuumio/miniconda3:4.7.12

# Conda env installation and setup based on code from:
# https://github.com/kaust-vislab/tensorflow-gpu-data-science-project/blob/0ef82814ec1cc00c1817d2fed4328fcec885f647/docker/Dockerfile
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env

RUN cat ~/.bashrc
RUN echo $PATH

COPY ./data_science/model_server/environment.yml ./environment.yml

SHELL [ "/bin/bash", "--login", "-c" ]

RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file ./environment.yml --force && \
    conda activate $ENV_PREFIX && \
    conda clean --all --yes

# make conda activate command for $ENV_PREFIX environment available from /bin/bash --interactive shells
RUN echo "source activate $ENV_PREFIX" > ~/.bashrc
RUN conda init bash
ENV PATH $ENV_PREFIX/bin:$PATH

COPY ./data_science ./data_science
COPY ./data_engineering ./data_engineering
COPY ./.gcs ./.gcs

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "data_science/model_server/main.py"]
