import json
import os


def get_model_and_metadata_from_gcs(bucket, model_dir, model_file_ext, model_load_func, gcs_model_dir, experiment_name):
    model_and_metadata_filepath = os.path.join(model_dir, experiment_name)
    metadata_filepath = f"{model_and_metadata_filepath}_metadata.json"
    model_filepath = f"{model_and_metadata_filepath}.{model_file_ext}"

    gcs_model_and_metadata_filepath = os.path.join(gcs_model_dir, experiment_name)
    gcs_metadata_filepath = f"{gcs_model_and_metadata_filepath}_metadata.json"
    gcs_model_filepath = f"{gcs_model_and_metadata_filepath}.{model_file_ext}"

    gcs_metadata_blob = bucket.blob(gcs_metadata_filepath)
    gcs_model_blob = bucket.blob(gcs_model_filepath)

    if gcs_metadata_blob.exists():
        print('Downloading model blob.')
        gcs_metadata_blob.download_to_filename(metadata_filepath)

        with open(metadata_filepath, 'r') as json_file:
            model_metadata = json.load(json_file)

        model_metadata['epochs_trained'] = int(model_metadata['epochs_trained'])

        gcs_model_blob.download_to_filename(model_filepath)

        model = model_load_func(model_filepath)
        return model, model_metadata

    return None, None