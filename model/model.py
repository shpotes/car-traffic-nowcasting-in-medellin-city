import model.layer
from model._model import _Model
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau


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

        model = Sequential([
            Conv2D(6, (5, 5), activation='tanh', name='C1', input_shape=size),
            AveragePooling2D((2, 2), name='S2'),
            Conv2D(16, (5, 5), activation='tanh', name='C3'),
            AveragePooling2D((2, 2), name='S4'),
            Conv2D(120, (5, 5), activation='tanh'),
            Flatten(),
            Dense(84, activation='tanh', name='F6'),
            Dense(num_classes, activation='softmax', name='output')])

        return model

class AlexNet(_Model):
    def callbacks(self):
        return [ReduceLROnPlateau()]

    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])

        model = Sequential([
            Conv2D(96, (11, 11), activation='relu', strides=4, 
                   name='CONV1', input_shape=size),
            MaxPooling2D((3, 3), strides=2, name='MAX_POOL1'),
            BatchNormalization(name='NORM1'),
            ZeroPadding2D((2, 2)),
            Conv2D(256, (5, 5), activation='relu', strides=1, name='CONV2'),
            MaxPooling2D((3, 3), strides=2, name='MAX_POOL2'),
            BatchNormalization(name='NORM2'),
            ZeroPadding2D((1, 1)),
            Conv2D(384, (3, 3), activation='relu', name='CONV3'),
            ZeroPadding2D((1, 1)),
            Conv2D(384, (3, 3), activation='relu', name='CONV4'),
            ZeroPadding2D((1, 1)),
            Conv2D(256, (3, 3), activation='relu', name='CONV5'),
            MaxPooling2D((3, 3), strides=2, name='MAX_POOL3'),
            Flatten(),
            Dense(4096, activation='relu', name='FC6'),
            Dropout(0.5),
            Dense(4096, activation='relu', name='FC7'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', name='FC8')])

        return model

class VGG16(_Model):
    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])
        
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=size),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            Conv2D(256, (3, 3), activation='relu'),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu'),
            Conv2D(512, (3, 3), activation='relu'),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu'),
            Conv2D(512, (3, 3), activation='relu'),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        return model

class ResNet(_Model):
    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])
        
        model = Sequential([Input(shape=size), 
                            Conv2D(64, (7, 7), strides=2),
                            MaxPooling2D((3, 3))])

        for i in range(4):
            for _ in range(2):
                model.add(layers.ResBlock(2 ** (5 + i)))
            if i < 4:
                model.add(layers.ResBlock(2 ** (5 + i), 
                                          bottleneck=2 ** (6 + i)))
            else:
                model.add(tf.keras.layers.GlobalAveragePooling2D())

        return model

class GoogleNet(_Model):
    def build_model(self):
        pass
