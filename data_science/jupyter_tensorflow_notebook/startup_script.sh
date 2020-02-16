#!/bin/bash

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "NVIDIA Docker installed"

# Check the driver until installed
while ! [[ -x "$(command -v nvidia-smi)" ]];
do
  echo "sleep to check"
  sleep 5s
done
echo "nvidia-smi is installed"

gcloud auth configure-docker
echo "Docker run with GPUs"
# Wait until disk is mounted
export MOUNTPOINT=YOUR_DISK_PATH
sudo mkdir -p $MOUNTPOINT
while [[ "$(lsblk -o MOUNTPOINT -nr /dev/sdb)" != $MOUNTPOINT ]]
do
  echo "waiting for disk to be attached to $MOUNTPOINT"
  sleep 5s
  sudo mount /dev/sdb $MOUNTPOINT
done
echo "Mounted disk at $MOUNTPOINT"

docker run -d --gpus all --log-driver=gcplogs \
-p 8888:8888 \
--volume $MOUNTPOINT:/home/jovyan/work \
us.gcr.io/$GCP_PROJECT/jupyter_tensorflow_notebook
echo "started notebook"