import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def build_source_from_metadata(metadata, data_dir, mode):
    df = metadata.copy().sample(frac=1).reset_index(drop=True)
    df = df[df['split'] == mode]
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, mode, x))

    sources = list(zip(df['filepath'], df['name'].apply(int)))
    return sources

def augment_image(img):
    return img

def load(raw):
    filepath = raw['image']
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img)
    return img, raw['label']

def make_dataset(sources, preprocess, training=False, batch_size=1,
               num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):

    if not shuffle_buffer_size:
        shuffle_buffer_size = batch_size * 4

    image, label = zip(*sources)

    ds = tf.data.Dataset.from_tensor_slices({
        'image' : list(image),
        'label' : list(label)
    })

    if training:
        ds.shuffle(shuffle_buffer_size)

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x, y: preprocess(x, y))

    if training:
        ds.map(lambda x, y: (augment_image(x), y))

    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def imshow_batch_of_three(batch, label_map=None, show_label=True):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()

    if not label_map:
        label_map = list(range(label_batch.max() + 1))

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        if show_label:
            axarr[i].set(xlabel='label = {}'.format(label_map[label_batch[i]]))

def imshow_with_predictions(model, batch, show_label=True, label_map=None):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()

    if not label_map:
        label_map = list(range(label_batch.max()))

    pred_batch = model.predict(image_batch)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        pred = int(np.argmax(pred_batch[i]))
        msg = f'pred = {pred}'
        if show_label:
            msg += f', label = {label_map[label_batch[i]]}'
        axarr[i].set(xlabel=msg)
