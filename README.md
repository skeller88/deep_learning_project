# Google Cloud Environment
Download `gcloud`.

Create auth key file and download to local machine. Copy to `big_earth_deep_learning_project/.gcs/credentials.json`.
The `.gcs` folder is gitignored.

```bash
export KEY_FILE=[your-key-file]
gcloud auth activate-service-account --key-file=$KEY_FILE
gcloud auth configure-docker

export PROJECT_ID=[your-project-id]
export HOSTNAME=us.gcr.io
export BASE_IMAGE_NAME=$HOSTNAME/$PROJECT_ID
```

# Download BigEarth archive to Google Cloud Storage bucket
## Run ETL image locally and deploy to google cloud
```
export SERVICE_NAME=archive_transferrer_from_big_earth_to_gcs
export FILEDIR=data_engineering/$SERVICE_NAME
export IMAGE_NAME=$BASE_IMAGE_NAME/$SERVICE_NAME
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile .
docker push $IMAGE_NAME
docker run -it --rm -p 8888:8888 \
--volume ~:/big-earth-data \
--env-file $FILEDIR/env.list $IMAGE_NAME
```

## Run on google cloud
```
# assumes persistent disk has not been created yet
gcloud compute instances create-with-container archive-transferrer \
        --zone=us-west1-b \
        --container-env-file=$FILEDIR/env.list \
        --container-image=$IMAGE_NAME \
        --container-mount-disk=name=big-earth-data,mount-path=/big-earth-data,mode=rw \
        --container-restart-policy=never \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --metadata-from-file=startup-script=startup_script.sh \
        --boot-disk-size=10GB \
        --create-disk=name=big-earth-data,auto-delete=no,mode=rw,size=200GB,type=pd-ssd,device-name=big-earth-data
```

# Data exploration and preparation
## Run ETL image locally and deploy to google cloud
```
export FILEDIR=data_engineering/archive_etler
export IMAGE_NAME=$BASE_IMAGE_NAME/archive_etler
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile .
docker push $IMAGE_NAME
docker run -it --rm -p 8888:8888 \
--volume ~:/big-earth-data \
--env-file $FILEDIR/env.list $IMAGE_NAME
```
Navigate to the `data_engineering/data_aggregator` folder for prototype notebooks.

## Run on google cloud
First time
```
gcloud compute instances create-with-container archive-etler \
        --zone=us-west1-b \
        --container-env-file=$FILEDIR/env.list \
        --container-image=$IMAGE_NAME \
        --container-mount-disk=name=big-earth-data,mount-path=/big-earth-data,mode=rw \
        --container-restart-policy=never \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --metadata-from-file=startup-script=startup_script.sh \
        --boot-disk-size=10GB \
        --disk=name=big-earth-data,auto-delete=no,mode=rw,device-name=big-earth-data
```

After the ETL job runs, download the metadata, the band statistics, and a few image files to your local machine for model prototyping.

```
export DISK_MOUNT_PATH=/mnt/disk
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/metadata/metadata.csv ~/data/metadata
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/metadata/band_stats.csv ~/data/metadata
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_45.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_46.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_47.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_48.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_49.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_50.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_51.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_52.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_53.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_54.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_55.npy ~/data/npy_image_files
gcloud compute scp archive-etler:$DISK_MOUNT_PATH/npy_image_files/S2A_MSIL2A_20170613T101031_0_56.npy ~/data/npy_image_files
```

Start and stop
```
gcloud compute instances stop archive-etler
gcloud compute instances start archive-etler

# ssh to instance and unmount disk
sudo umount /dev/disk/by-id/google-big-earth-data

# stop instance and detach the disk when done with ETL
gcloud compute instances stop archive-etler
gcloud compute instances detach-disk archive-etler --disk=big-earth-data
```

# Model training
## Prototype locally with Jupyter notebook
```
# Prepare a hashed password:
# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password
export JUPYTER_PASSWORD_SHA=[your-hashed-password-from-above-step]
export FILEDIR=data_science/jupyter_tensorflow_notebook
export IMAGE_NAME=$BASE_IMAGE_NAME/jupyter_tensorflow_notebook
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile --build-arg jupyter_password_sha_build_arg=$JUPYTER_PASSWORD_SHA .
docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work $IMAGE_NAME
docker push $IMAGE_NAME
```

## Create GCP instance from Google image family
```
# scopes needed are pub/sub, service control, service management, container registry,
# stackdriver logging/trace/monitoring, storage
# Full names: --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/logging.admin,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/source.read_only \
export DISK_NAME=disk
export DISK_MOUNT_PATH=/mnt/disk
export FILEDIR=data_science/jupyter_tensorflow_notebook
export IMAGE_PROJECT=deeplearning-platform-release
export IMAGE_FAMILY=common-cu100
gcloud compute instances create jupyter-tensorflow-notebook \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-8 \
        --boot-disk-size=50GB \
        --metadata=enable-oslogin=TRUE,install-nvidia-driver=True \
        --metadata-from-file=startup-script=$FILEDIR/startup_script.sh \
        --disk=name=$DISK_NAME,auto-delete=no,mode=rw,device-name=$DISK_NAME \
        --tags http-server

# Upload model notebook from local to instance
gcloud compute scp ~/Documents/big_earth_deep_learning_project/data_science/model.ipynb jupyter-tensorflow-notebook:$DISK_MOUNT_PATH

# Navigate to the instance IP address and login to the notebook. There should be a jupyter-tensorflow-notebook/model.ipynb
# notebook file in the home directory. Open the notebook and run it to train the models.

# If you make changes to the model.ipynb notebook while running it on the instance, download the notebook from the
# instance to your local machine to sync changes.
gcloud compute scp jupyter-tensorflow-notebook:$DISK_MOUNT_PATH/model.ipynb ~/Documents/big_earth_deep_learning_project/data_science
```

# Model deployment
```
export FILEDIR=data_science/model_server
export IMAGE_NAME=$BASE_IMAGE_NAME/model_server
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile  .
docker run -it --rm --env-file $FILEDIR/env.list -p 8889:8889 $IMAGE_NAME
docker push $IMAGE_NAME

gcloud compute instances create-with-container model-server \
        --zone=us-west1-b \
        --can-ip-forward \
        --container-image=$IMAGE_NAME \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-1 \
        --boot-disk-size=1GB \
        --metadata enable-oslogin=TRUE \
        --tags http-server

## make request
python data_science/model_server/make_test_request.py
```
