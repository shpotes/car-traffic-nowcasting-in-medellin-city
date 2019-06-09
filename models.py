import tensorflow as tf

# TODO: LeNet
# TODO: AlexNet
# TODO: VGG16
# TODO: ResNet

def linear_model(size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=size),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizer.Adam(lr=3e-4),
        metrics=['accuracy']
    )

    return model
