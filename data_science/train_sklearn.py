import json
import os
import time

import numpy as np
import joblib
import datetime

import sklearn
from joblib import dump
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

from data_science.serialization_utils import numpy_to_json, sklearn_precision_recall_curve_to_dict
from data_science.train import get_model_and_metadata_from_gcs


def train_sgd_classifier(random_seed, bucket, model_dir, train_batch_generator, valid_batch_generator, n_epochs):
    history = list()

    classes = np.array([0, 1])
    epochs_without_improvement = 0

    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    experiment_name = f"sgd_classifier_default_2020_1_31"
    gcs_model_dir = "model/models"
    model_path = os.path.join(model_dir, experiment_name + ".joblib")
    model_gcs_path = os.path.join(gcs_model_dir, experiment_name + ".joblib")
    model_metadata_path = os.path.join(model_dir, experiment_name + "_metadata.json")
    model_metadata_gcs_path = os.path.join(gcs_model_dir, experiment_name + "_metadata.json")

    model, model_base_metadata = get_model_and_metadata_from_gcs(bucket, model_dir, "joblib", joblib.load, gcs_model_dir,
                                                                 experiment_name)

    if model is not None:
        print('Resuming training at epoch', model_base_metadata['epoch'])
    else:
        model = SGDClassifier(loss='log')
        model_base_metadata = {
            'data': 'train_valid_google_automl_cloud_and_shadow_dataset_small.csv',
            'data_prep': 'normalization_augmentation',
            'experiment_name': experiment_name,
            'experiment_start_time': now,
            'model': SGDClassifier.__name__,
            'random_state': random_seed,
            'epoch': 0
        }

    # Shuffle the data
    train_batch_generator.on_epoch_end()
    valid_batch_generator.on_epoch_end()
    train_start = time.time()
    best_model = None
    max_accuracy_valid = None
    early_stopping_patience = 15
    for epoch in range(int(model_base_metadata['epoch']) + 1, n_epochs):
        start = time.time()
        for batch_x, batch_y in train_batch_generator.make_one_shot_iterator():
            model.partial_fit(batch_x, batch_y, classes=classes)

        if epoch % 10 == 0:
            print("training completed in", time.time() - start, "seconds")

        start = time.time()

        actual_y_train, pred_y_train = train_batch_generator.get_predictions(model)
        actual_y_valid, pred_y_valid = valid_batch_generator.get_predictions(model)

        if epoch % 10 == 0:
            print("prediction completed in", time.time() - start, "seconds")

        epoch_time = f"{time.time() - start:.4f}"
        epoch_metrics = {
            'accuracy_train': sklearn.metrics.accuracy_score(actual_y_train, pred_y_train),
            'accuracy_valid': sklearn.metrics.accuracy_score(actual_y_valid, pred_y_valid),
            "f1_score_train": sklearn.metrics.f1_score(actual_y_train, pred_y_train),
            "f1_score_valid": sklearn.metrics.f1_score(actual_y_valid, pred_y_valid),
        }
        history.append(epoch_metrics)

        print("epoch_num", epoch, "-", epoch_time, "sec -", epoch_metrics['accuracy_valid'])

        if max_accuracy_valid is None or epoch_metrics['accuracy_valid'] > max_accuracy_valid:
            max_accuracy_valid = epoch_metrics['accuracy_valid']
            dump(model, model_path)
            with open(model_metadata_path, 'w+') as json_file:
                model_base_metadata.update({
                    'epoch': str(epoch),
                    'confusion_matrix': numpy_to_json(confusion_matrix(actual_y_valid, pred_y_valid)),
                    'precision_recall_curve': sklearn_precision_recall_curve_to_dict(
                        sklearn.metrics.precision_recall_curve(actual_y_valid, pred_y_valid)),
                    'history': history,
                    'train_time_elapsed': time.time() - train_start
                })
                json.dump(model_base_metadata, json_file)

            for filename, gcs_filename in [(model_path, model_gcs_path), (model_metadata_path, model_metadata_gcs_path)]:
                blob = bucket.blob(gcs_filename)
                blob.upload_from_filename(filename)

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == early_stopping_patience:
            print("Ending training due to no improvement")
            break

        train_batch_generator.on_epoch_end()
        valid_batch_generator.on_epoch_end()
