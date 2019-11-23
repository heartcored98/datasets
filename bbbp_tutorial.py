import tensorflow_datasets as tfds
import tensorflow as tf

# Show bbbp dataset info
bbbp_builder = tfds.builder("bbbp")
print(bbbp_builder.info)

# Very simple way to create/prepare/split bbbp dataset
dataset = tfds.load(name="bbbp", split=tfds.Split.TRAIN)

dataset = dataset.shuffle(128).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for features in dataset.take(1):
    smile, label = features["smile"], features["label"]
    print(smile, label)

Sentence encoders based on the trans-
former architecture have shown promising
results on various natural language tasks.
The main impetus lies in the pre-trained
neural language models that capture long-
range dependencies among words, owing
to multi-head attention that is unique in the
architecture. However, little is known for how linguistic properties are processed, represented and utilized for downstream tasks among hundreds of attention heads
inside the pre-trained transformer model.
For the first goal of examining the roles of
attention heads in handling a set of
linguistic features, we conducted a set of experiments with ten probing tasks and
three downstream tasks on four recently
developed pre-trained transformer en-
coders. Meaningful insights are shown through the lens of heat map visualization and utilized to propose a relatively simple sentence representation method that takes
advantage of most influential attention
heads, resulting in significant performance
improvements on the downstream tasks.