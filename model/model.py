import tensorflow as tf
from model.layer import ResBlock, Inception
from model.callbacks import custom_calls
from model._model import _Model
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout



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
    def callbacks(self, callbacks):
        return custom_calls(self, callbacks)

    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])
        
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu',
                   padding='same', input_shape=size),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        return model

class ResNet(_Model):
    def build_model(self, N=3, rep=2, L=6):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])
        
        model = Sequential([Input(shape=size), 
                            ZeroPadding2D((3, 3)),
                            Conv2D(2 ** L, (7, 7), strides=2,
                                   activation='relu',
                                   kernel_initializer='he_normal'),
                            ZeroPadding2D((1, 1)),
                            MaxPooling2D((3, 3), strides=2)])
        for i in range(N):
            for _ in range(rep):
                model.add(ResBlock(2 ** (L + i)))
            model.add(ResBlock(2 ** (L + i + 1), bottleneck=2 ** (L + i + 1)))
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(2 ** (L + i + 1), (3,3), strides=2,
                             activation='relu', kernel_initializer='he_normal'))

        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', 
                        kernel_initializer='he_normal'))
        return model

class GoogleNet(_Model):
    def preprocess(self, img, label):
        size = self.config['model']['input_size']
        num_classes = len(self.config['model']['labels'])
        img = tf.image.resize(img, size)
        img /= 255.0
        y = tf.one_hot(indices=label, depth=num_classes)
        return img, (y, y, y)

    def build_model(self):
        size = self.config['model']['input_size'] + [3]
        num_classes = len(self.config['model']['labels'])
        
        inputs = Input(shape=size)
        x = ZeroPadding2D((3, 3))(inputs)
        x = Conv2D(64, (7, 7), strides=2, activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(192, (1, 1), strides=1, activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = Inception([64, 128, 32], [None, 96, 16],
                      pool_proj=32, name='inc_3a')(x)
        x = Inception([128, 192, 96], [None, 96, 16],
                      pool_proj=64, name='inc_3b')(x)

        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x, out1 = Inception([192, 208, 48], [None, 96, 16],
                            pool_proj=64, name='inc_4a',
                            num_classes=num_classes)(x)    
        x = Inception([160, 224, 64], [None, 112, 24],
                      pool_proj=64, name='inc_4b')(x)    
        x = Inception([128, 256, 64], [None, 128, 24],
                      pool_proj=64, name='inc_4c')(x)    

        x, out2 = Inception([112, 288, 64], [None, 114, 32],
                            pool_proj=64, name='inc_4d', 
                            num_classes=num_classes)(x)
        x = Inception([256, 320, 128], [None, 160, 32],
                      pool_proj=128, name='inc_4e')(x)

        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = Inception([256, 320, 128], [None, 160, 32],
                      pool_proj=128, name='inc_5a')(x)
        x = Inception([384, 384, 128], [None, 192, 48],
                      pool_proj=128, name='inc_5b')(x)
        
        x = AveragePooling2D((7, 7), strides=1)(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        
        output = Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs, [out1, out2, output])
