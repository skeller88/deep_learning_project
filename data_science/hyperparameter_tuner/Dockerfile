# Tensorflow image Dockerfile:
# https://github.com/tensorflow/tensorflow/blob/d73faf5fbb7c8bfbf96cc6334111e4352f209e82/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
# FROM continuumio/miniconda:4.7.12

# Image Dockerfile:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/ubuntu18.04/10.0/runtime/Dockerfile
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# https://askubuntu.com/questions/141928/what-is-the-difference-between-bin-sh-and-bin-bash
# bash has more functionality than bin, such as "source"
SHELL [ "/bin/bash", "--login", "-c" ]

# From https://github.com/ContinuumIO/docker-images/blob/master/miniconda/debian/Dockerfile
# Changed to not activate base environment from .bashrc
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean


ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    # make conda activate command available from /bin/bash --login shells
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find $CONDA_DIR/ -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR/ -follow -type f -name '*.js.map' -delete && \
    $CONDA_DIR/bin/conda clean -afy

# make non-activate conda commands available
ENV PATH $CONDA_DIR/bin:$PATH

# Conda env installation and setup based on code from:
# https://github.com/kaust-vislab/tensorflow-gpu-data-science-project/blob/0ef82814ec1cc00c1817d2fed4328fcec885f647/docker/Dockerfile
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env

RUN cat ~/.bashrc
RUN echo $PATH

COPY ./data_science/hyperparameter_tuner/environment.yml ./environment.yml

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

RUN mkdir /mnt/big-earth-data

# RUN mkdir -p /host_mnt
# RUN chown newuser /host_mnt
# USER newuser

# Allow python to discover modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Tensorflow requires this: https://www.tensorflow.org/install/gpu
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
RUN echo $LD_LIBRARY_PATH

CMD ["python", "data_science/hyperparameter_tuner/main.py"]
