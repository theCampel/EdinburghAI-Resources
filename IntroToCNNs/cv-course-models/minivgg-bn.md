---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# MiniVGG BatchNorm #

```python
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
```

```python
import visiontools
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
```

## Untrained ##

```python
SIZE = [192, 192]

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block1_conv1", input_shape=[*SIZE, 3]),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(name="block1_pool"),
    layers.Conv2D(filters=128,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block2_conv1"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(name="block2_pool"),
    layers.Conv2D(filters=256,
                  kernel_size=3,
                  activation="relu",
                  padding='same',
                  name="block3_conv1"),
    layers.Conv2D(filters=256,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block3_conv2"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(name="block3_pool"),
    layers.Conv2D(filters=512,
                  kernel_size=3,
                  activation="relu",
                  padding='same',
                  name="block4_conv1"),
    layers.Conv2D(filters=512,
                  kernel_size=3,
                  activation="relu",
                  padding='same',
                  name="block4_conv2"),
    layers.Conv2D(filters=512,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block4_conv3"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(name="block4_pool"),
    layers.Flatten(),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
], name='minivgg')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
model.save('minivgg-bn-untrained')
```

## Pretrained ##

```python
# Load training and validation sets
DATA_DIR = '/home/jovyan/work/kaggle/datasets'
(ds_train_, ds_valid_), ds_info = tfds.load('stanford_cars/simple',
                                          split=['train', 'test'],
                                          shuffle_files=True,
                                          with_info=True,
                                          data_dir=DATA_DIR)

BATCH_SIZE = 16
AUTO = tf.data.experimental.AUTOTUNE
SIZE = [192, 192]
preprocess = visiontools.make_preprocessor(size=SIZE)

ds_train = (ds_train_
            .map(preprocess, AUTO)
            .cache()
            .repeat()
            .shuffle(ds_info.splits['train'].num_examples)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

ds_valid = (ds_valid_
            .map(preprocess, AUTO)
            .cache()
            .shuffle(ds_info.splits['test'].num_examples)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))
```

```python
early_stopping = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=2)

EPOCHS = 100
STEPS_PER_EPOCH = ds_info.splits['train'].num_examples // BATCH_SIZE

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stopping, lr_schedule],
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

```python
model.save('minivgg-bn-trained')
```
