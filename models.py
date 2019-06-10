import json

import tensorflow as tf
import pandas as pd

from utils import build_source_from_metadata, make_dataset
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, AveragePooling2D, Input

# TODO: LeNet
# TODO: AlexNet
# TODO: VGG16
# TODO: ResNet

class _Model:
    def __init__(self, config):
        self.config = config

        self.metadata = pd.read_csv(self.config['model']['metadata_path'])
        self.model = self.build_model()

    def __str__(self):
        string = [self.__class__.__name__, '']
        self.model.summary(print_fn=lambda x: string.append(x))
        return '\n'.join(string)

    def build_model(self):
        pass

    def preprocess(self, img, label):
        size = self.config['model']['input_size']
        num_classes = len(self.config['model']['labels'])
        img = tf.image.resize(img, size)
        img /= 255.0
        y = tf.one_hot(indices=label, depth=num_classes)
        return img, y

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def _create_data(self, overfit_mode, test=False):
        BATCH_SIZE = self.config['train']['batch_size']

        if overfit_mode:
            metadata = self.metadata[self.metadata.split=='train']
            metadata = metadata.groupby('name', as_index=False).apply(
                lambda x: x.sample(1)
            ).reset_index(drop=True)
            BATCH_SIZE = len(self.config['model']['labels'])
        else:
            metadata = self.metadata

        train_source = build_source_from_metadata(
            metadata,
            self.config['model']['data_path'],
            'train'
        )

        train_data = make_dataset(train_source,
                                  training=True,
                                  batch_size=BATCH_SIZE,
                                  num_parallel_calls=8,
                                  preprocess=lambda x, y: self.preprocess(x, y))

        if test:
            train_source = build_source_from_metadata(
                self.metadata,
                self.config['model']['data_path'],
                'test'
            )
            test_data = make_dataset(train_source,
                                     training=False,
                                     batch_size=BATCH_SIZE,
                                     num_parallel_calls=8,
                                     preprocess=lambda x, y: self.preprocess(x, y))

            return train_data, test_data
        return train_data

    def train(self, overfit_mode=False):
        VAL_SPLIT = self.config['train']['validation_split']
        LR = self.config['train']['learning_rate']
        NUM_EPOCHS = self.config['train']['over_epochs'] if overfit_mode \
            else self.config['train']['num_epochs']
        METRICS = self.config['train']['metrics']
        CALLBACKS = self.config['train']['callbacks']

        if not CALLBACKS:
            self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                               optimizer=tf.optimizers.Adam(LR),
                               metrics=METRICS)
        else:
            from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

            CALLBACKS = [EarlyStopping(patience=0),
                         TensorBoard(),
                         ReduceLROnPlateau(),
                         ModelCheckpoint('chpts/w.%s.{epoch:02d}-{val_loss:.2f}.h5') %  self.__class__.__name__]

            self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                               optimizer=tf.optimizers.Adam(LR),
                               metrics=METRICS,
                               callbacks=CALLBACKS)

        train_data = self._create_data(overfit_mode)
        self.model.fit(train_data, epochs=NUM_EPOCHS)

    def evaluate(self, overfit_mode=False):
        train_data, test_data = self._create_data(overfit_mode, test=True)

        print('train')
        self.model.evaluate(train_data)
        print('test')
        self.model.evaluate(test_data)

    def predict(self, ds):
        # TODO: Add preprocessing
        return self.model.predict(ds)

class LinearModel(_Model):
    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])

        model = Sequential([
            Flatten(input_shape=size),
            Dense(num_classes, activation='softmax')
        ])

        return model

class LeNet(_Model):
    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])

        INPUT = Input(shape=size)
        C1 = Conv2D(6, (5, 5), activation='tanh', name='C1')(INPUT)
        S2 = AveragePooling2D((2, 2), name='S2')(C1)
        C3 = Conv2D(16, (5, 5), activation='tanh', name='C3')(S2)
        S4 = AveragePooling2D((2, 2), name='S4')(C3)
        C5 = Conv2D(120, (5, 5), activation='tanh')(S4)
        C5 = Flatten()(C5)
        F6 = Dense(84, activation='tanh', name='F6')(C5)
        OUTPUT = Dense(num_classes, activation='softmax', name='output')(F6)

        model = Model(INPUT, OUTPUT)
        return model

class VGG16(_Model):
    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])

        INPUT = Input(shape=size)
        # TODO: VGG Like here
        OUTPUT = Dense(num_classes, activation='softmax', name='output')(x)

        model = Model(INPUT, OUTPUT)
        return model
