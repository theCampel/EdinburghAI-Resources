---
jupyter:
  jupytext:
    formats: md,ipynb
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

# VGG 16 #

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
```

## Pretrained Base ##

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

augment = visiontools.make_augmentor(horizontal_flip=True,
                         brightness_delta=0.05,
                         hue_delta=0.2,
                         saturation_range=[0.05, 1.15],
                         contrast_range=[0.8, 1.2])

ds_train = (ds_train_
            .map(preprocess, AUTO)
            .cache()
            .repeat()
            # .map(augment, AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

ds_valid = (ds_valid_
            .map(preprocess, AUTO)
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(AUTO))
```

```python

```
