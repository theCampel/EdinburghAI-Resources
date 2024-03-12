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

# Simple CNN #

```python
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
```

## Untrained ##

```python
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers

model = Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block1_conv1"),
    layers.BatchNormalization(),
    layers.Activation('relu')
    layers.MaxPool2D(name="block1_pool"),
    layers.Conv2D(filters=128,
                  kernel_size=3,
                  activation=None,
                  padding='same',
                  name="block2_conv1"),
    layers.BatchNormalization(),
    layers.Activation('relu')
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
    layers.Activation('relu')
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
    layers.Activation('relu')
    layers.MaxPool2D(name="block4_pool"),
    layers.Flatten(),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
], name='minivgg')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.save('minivgg-untrained')
```

## Pretrained ##

```python
early_stopping = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=2)

EPOCHS = 100
STEPS_PER_EPOCH = ds_info.splits['train'].num_examples

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stopping, lr_schedule],
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();
```

```python
model.save('minivgg-trained')
```
