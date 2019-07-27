import tensorflow as tf
from tensorflow.keras.layers import add, Layer, BatchNormalization, Conv2D
from tensorflow.keras.layers import Concatenate, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dropout, Dense

class ResBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), bottleneck=0):
        super(ResBlock, self).__init__()
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters, kernel_size, padding='same',
                            use_bias=False, activation='relu', 
                            kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size, padding='same',
                            use_bias=False, activation='relu',
                            kernel_initializer='he_normal')
        
        if bottleneck:
            self.bottleneck = Conv2D(bottleneck, (1, 1), activation='relu',
                                     kernel_initializer='he_normal')
        else:
            self.bottleneck = None

    def call(self, inputs):
        shortcut = inputs
        
        x = self.bn1(inputs)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if self.bottleneck:
            shortcut = self.bottleneck(shortcut)

        x = add([shortcut, x])
        return x

class Inception(Layer):
    def __init__(self, filter_list, filters_reduce, kernel_list=[1, 3, 5],
                 pool_proj=0, batch_norm=False, num_classes=0, name=''):
        
        super(Inception, self).__init__()
        
        kernel_list = zip(kernel_list, kernel_list)
        self.num_classes = num_classes
        self.convs = [Conv2D(filters, kernel_size, activation='relu', padding='same') 
                      for filters, kernel_size in zip(filter_list, kernel_list)]
        self.red = [Conv2D(filters, (1, 1), activation='relu', padding='same') if filters else None
                    for filters in filters_reduce]
        if pool_proj:
            self.red.append(MaxPooling2D((3, 3), strides=1, padding='same'))
            self.convs.append(Conv2D(pool_proj, (1, 1), activation='relu'))

        if self.num_classes:
            self.fc = [AveragePooling2D((5, 5), strides=3),
                       Conv2D(128, (1, 1), activation='relu'),
                       Dense(1000, activation='relu'),
                       Dropout(0.7),
                       Dense(num_classes, activation='softmax')]
        self.batch_norm = batch_norm # TODO
        self.concat = Concatenate()

        

    def call(self, inputs):
        reduction = [conv(inputs) if conv else inputs for conv in self.red]
        hebbian = [conv(red) for conv, red in zip(self.convs, reduction)]
        
        if self.num_classes:
            x = inputs
            for step in self.fc:
                x = step(x)
            output = x
            return self.concat(hebbian), output
        
        return self.concat(hebbian)
