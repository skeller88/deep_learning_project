import random
import time

import numpy as np


class SklearnBatchGenerator:
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, band_stats, has_verbose_logging=False,
                 should_test_time_augment=False):
        self.x = x
        self.y = y
        self.base_index = [idx for idx in range(len(x))]
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.has_verbose_logging = has_verbose_logging
        self.means = band_stats['mean'].values
        self.stds = band_stats['std'].values
        self.should_test_time_augment = should_test_time_augment

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, batch_num):
        img_names = self.x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        if self.y is not None:
            batch_y = np.ravel(self.y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size])

            start = time.time()
            batch_x = self.batch_loader(img_names, self.augmentations is not None)

            if batch_num == 0 and self.has_verbose_logging:
                print('fetched batch_num', batch_num, 'in', time.time() - start, 'seconds')

            return batch_x, batch_y
        # test (inference only)
        else:
            return self.batch_loader(img_names, self.should_test_time_augment)

    def make_one_shot_iterator(self):
        for batch_num in range(len(self)):
            batch_x, batch_y = self[batch_num]
            yield batch_x, batch_y

    def batch_loader(self, image_paths, should_augment) -> np.array:
        imgs = np.array([np.load(image_path) for image_path in image_paths])
        normalized_imgs = (imgs - self.means) / self.stds

        if should_augment:
            return np.stack([self.augmentations(image=x)["image"].flatten() for x in normalized_imgs], axis=0)
        return np.array([x.flatten() for x in normalized_imgs])

    def on_epoch_end(self):
        # Can't figure out how to pass in the epoch variable
        #         print_with_stdout(f"epoch {epoch}, logs {logs}")
        #         if logs is not None:
        #             print_with_stdout(f"finished epoch {epoch} with accuracy {logs['acc']} and val_loss {logs['val_loss']}")
        #         else:
        #             print_with_stdout(f"finished epoch {epoch}")
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        shuffled_index = self.base_index.copy()
        random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]

        if self.y is not None:
            self.y = self.y[shuffled_index]

    def get_predictions(self, model):
        pred_y_batches = []
        actual_y_batches = []
        for batch_x, batch_y in self.make_one_shot_iterator():
            pred_y_batches.append(model.predict(batch_x))
            actual_y_batches.append(batch_y)

        pred_y = []
        for pred_y_batch in pred_y_batches:
            for pred in pred_y_batch:
                pred_y.append(pred)

        actual_y = []
        for actual_y_batch in actual_y_batches:
            for actual in actual_y_batch:
                actual_y.append(actual)

        return actual_y, pred_y


