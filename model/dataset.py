import os
import numpy as np
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

    label_map = np.array(label_map)

    pred_batch = model.predict(image_batch)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        pred = int(np.argmax(pred_batch[i]))
        msg = 'pred = %s' % label_map[pred]
        if show_label:
            msg += ', label = %s' % label_map[label_batch[i].astype(bool)]
        axarr[i].set(title=msg)
        axarr[i].axis('off')

def compute_confusion_matrix(model):
    return tf.math.confusion_matrix(model.y_true, model.y_pred)
    

def create_data(model, overfit_mode, test=False):
    BATCH_SIZE = model.config['train']['batch_size']

    if overfit_mode:
        metadata_train = model.metadata[model.metadata.split == 'train']
        metadata_train = metadata_train.groupby('name', as_index=False).apply(
            lambda x: x.sample(1)
        ).reset_index(drop=True)
        metadata = model.metadata[model.metadata.split != 'train']
        BATCH_SIZE = len(model.config['model']['labels'])
        N_EPOCHS = model.config['train']['num_epochs']

    else:
        metadata = model.metadata
        metadata_train = model.metadata
        
    train_source = build_source_from_metadata(
        metadata_train,
        model.config['model']['data_path'],
        'train'
    )

    train_data = make_dataset(train_source,
                              training=True,
                              batch_size=BATCH_SIZE,
                              num_parallel_calls=8,
                              preprocess=lambda x, y: model.preprocess(x, y))

    test_source = build_source_from_metadata(
        metadata,
        model.config['model']['data_path'],
        'test'
    )
    test_data = make_dataset(test_source,
                             training=False,
                             batch_size=BATCH_SIZE,
                             num_parallel_calls=8,
                             preprocess=lambda x, y: model.preprocess(x, y))
    
    val_source = build_source_from_metadata(
        metadata,
        model.config['model']['data_path'],
        'val'
    )

    val_data = make_dataset(val_source,
                             training=False,
                             batch_size=BATCH_SIZE,
                             num_parallel_calls=8,
                             preprocess=lambda x, y: model.preprocess(x, y))

    return train_data, val_data, test_data
