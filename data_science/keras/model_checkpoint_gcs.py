import json
import time
import warnings

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint


class ModelCheckpointGCS(ModelCheckpoint):
    """
    Computes scikit-learn metrics on train and validation data whenever model reaches a new high "monitor" value. Saves
    model and training metadata to disk and gcs. Assumes GOOGLE_APPLICATION_CREDENTIALS has been set.
    """

    def __init__(self, filepath, gcs_filepath, gcs_bucket, model_metadata, monitor='val_loss', verbose=0, mode='auto',
                 period=1):
        model_filepath = f"{filepath}.h5"
        super(ModelCheckpointGCS, self).__init__(filepath=model_filepath, monitor=monitor, verbose=verbose,
                                                 save_best_only=True, save_weights_only=False,
                                                 mode=mode, period=period)
        self.model_filepath = model_filepath
        self.model_metadata_filepath = f"{filepath}_metadata.json"
        self.gcs_bucket = gcs_bucket
        self.gcs_model_filepath = f"{gcs_filepath}.h5"
        self.gcs_model_metadata_filepath = f"{gcs_filepath}_metadata.json"
        self.model_metadata = model_metadata
        self.train_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Based on
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/keras/callbacks.py#L983
        :param epoch:
        :param logs:
        :return:
        """
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.model_filepath))
                self.best = current

                # Save model
                self.model.save(self.model_filepath, overwrite=True)

                blob = self.gcs_bucket.blob(self.gcs_model_filepath)
                blob.upload_from_filename(self.model_filepath)

                self.model_metadata.update({
                    'epoch': str(epoch),
                    'history': {key: value.astype(np.float64) for key, value in logs.items()},
                    'elapsed_train_time': time.time() - self.train_start_time
                })

                with open(self.model_metadata_filepath, 'w+') as json_file:
                    json.dump(self.model_metadata, json_file)

                blob = self.gcs_bucket.blob(self.gcs_model_metadata_filepath)
                blob.upload_from_filename(self.model_metadata_filepath)

            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' %
                          (epoch, self.monitor))
