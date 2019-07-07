import tensorflow as tf
from tensorflow.keras.layers import add, Layer, BatchNormalization, Conv2D

class ResBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), bottleneck=0):
        super(ResBlock, self).__init__()
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        
        if bottleneck:
            self.bottleneck = Conv2D(bottleneck, (1, 1), padding='same')
        else:
            bottleneck = None

    def call(self, inputs):
        shortcut = inputs
        
        x = self.bn1(inputs)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if bottleneck:
            self.bottleneck(shortcut)

        x = add([shortcut, x])
        return x
