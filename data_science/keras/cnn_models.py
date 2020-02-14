# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model


def basic_cnn_model(img_shape, n_classes):
    """
    From https://arxiv.org/pdf/1902.06148.pdf

    To this end, we selected a shallow CNN architecture, which consists of three convolutional layers with 32, 32 and
    64 filters having 5 × 5, 5 × 5 and 3 × 3 filter sizes, respectively. We
    added one fully connected (FC) layer and one classification
    layer to the output of last convolutional layer. In all convolution operations, zero padding was used. We also applied
    max-pooling between layers.
    """
    kernel_initializer = 'he_uniform'
    img_inputs = Input(shape=img_shape)
    conv_1 = Conv2D(32, (5, 5), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(img_inputs)
    maxpool_1 = MaxPooling2D((2, 2))(conv_1)
    conv_2 = Conv2D(32, (5, 5), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(maxpool_1)
    maxpool_2 = MaxPooling2D((2, 2))(conv_2)
    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(maxpool_2)
    flatten = Flatten()(conv_3)
    dense_1 = Dense(64, activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(flatten)
    output = Dense(n_classes, activation='sigmoid')(dense_1)

    return Model(inputs=img_inputs, outputs=output)


def basic_cnn_model_with_regularization(img_shape, n_classes):
    """
    From https://arxiv.org/pdf/1902.06148.pdf

    To this end, we selected a shallow CNN architecture, which consists of three convolutional layers with 32, 32 and
    64 filters having 5 × 5, 5 × 5 and 3 × 3 filter sizes, respectively. We
    added one fully connected (FC) layer and one classification
    layer to the output of last convolutional layer. In all convolution operations, zero padding was used. We also applied
    max-pooling between layers.

    Add batch normalization and appropriate weight initialization.
    """
    kernel_initializer = 'he_uniform'
    img_inputs = Input(shape=img_shape)
    conv_1 = Conv2D(32, (5, 5), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(img_inputs)
    bn_1 = BatchNormalization()(conv_1)
    maxpool_1 = MaxPooling2D((2, 2))(bn_1)
    conv_2 = Conv2D(32, (5, 5), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(maxpool_1)
    bn_2 = BatchNormalization()(conv_2)
    maxpool_2 = MaxPooling2D((2, 2))(bn_2)
    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(maxpool_2)
    bn_3 = BatchNormalization()(conv_3)
    flatten = Flatten()(bn_3)
    dense_1 = Dense(64, activation='relu', kernel_initializer=kernel_initializer, use_bias=False)(flatten)
    bn_dense_1 = BatchNormalization()(dense_1)
    output = Dense(n_classes, activation='sigmoid')(bn_dense_1)

    return Model(inputs=img_inputs, outputs=output)


def pretrained_model(base_model_class, input_shape, output_shape):
    """
    All of the top performers use transfer learning and image augmentation: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/33559

    Another useful discussion on both topics: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36091#202629
    """
    # from https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93#L321
    base_model = base_model_class(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    base_output = base_model.output
    gap = GlobalAveragePooling2D()(base_output)
    dense = Dense(2048, activation='relu')(gap)
    dropout = Dropout(0.25)(dense)
    output = Dense(output_shape, activation='sigmoid')(dropout)
    model = Model(inputs=base_model.inputs, outputs=output)

    return model
