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
from tensorflow.keras.applications import VGG16

pretrained_base = VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=[192, 192, 3])
pretrained_base.save('vgg16-pretrained-base')
```
