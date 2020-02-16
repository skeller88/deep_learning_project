from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def join_histories(histories):
    full_history = defaultdict(list)

    for history in histories:
        for key, value in history.history.items():
            full_history[key].extend(value)
    return full_history


def graph_model_history(history, title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle(title, fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    max_epoch = len(history['val_loss']) + 1
    epoch_list = list(range(1, max_epoch))
    ax1.plot(epoch_list, history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")