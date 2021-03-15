import numpy as np
import pandas as pd

import thinc
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization  # to create AdamW optmizer


from preprocessing import preprocess_dataset as clean_data

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42
cat_codes = {"neg": 0, "neu": 1, "pos": 2, "irr": 3}


def train_val_test_split(df, train_size=0.8, val_size=0.1):
    return np.split(
        df.sample(frac=1, random_state=seed),
        [int(train_size * len(df)), int((1 - val_size) * len(df))],
    )


def load_data(path="data/df.pk"):
    data = clean_data(pd.read_pickle(path), fasttext_model_location="data/lid.176.bin")
    data = data[data.Avis != "irr"]
    data["avis"] = data.Avis.map(cat_codes.get)
    data.Tweet = lemmatize(data.Tweet)

    train, val, test = train_val_test_split(data)

    print(train, val, test)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (train.Tweet.to_numpy(), train.avis.to_numpy(dtype=np.int32))
        )
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (val.Tweet.to_numpy(), val.avis.to_numpy(dtype=np.int32))
        )
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(
            (test.Tweet.to_numpy(), test.avis.to_numpy(dtype=np.int32))
        )
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    return train_ds, val_ds, test_ds


def tokenize(tweets: pd.Series) -> pd.Series:
    return tweets


def lemmatize(tweets):
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    return [
        " ".join(w.lemma_ for w in doc if not w.is_stop)
        for doc in nlp.pipe(tweets, batch_size=50)
    ]


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="preprocessing",
    )
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1",
        trainable=True,
        name="BERT_encoder",
    )
    outputs = encoder(encoder_inputs)

    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation="relu", name="layer1")(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(3, activation="relu", name="layer2")(net)
    return tf.keras.Model(text_input, net)


def train(classifier_model, train_ds, val_ds):
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.SparseCategoricalAccuracy()

    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = classifier_model.fit(x=train_ds, validation_data=val_ds, epochs=epochs)
    return history


def evaluate(classifier_model, test_ds):
    loss, accuracy = classifier_model.evaluate(test_ds)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


def predict(tweets, model_path="data/twitter_bert"):

    model = None

    import os

    if os.path.exists(model_path):
        model = tf.saved_model.load(model_path)

    if model is not None:
        return np.argmax(model(tf.constant(tweets)).numpy(), axis=1)

    return


if __name__ == "__main__":
    # train_ds, val_ds, test_ds = load_data()
    #
    # classifier_model = build_classifier_model()
    # train(classifier_model, train_ds, val_ds)
    # evaluate(classifier_model, test_ds)
    #
    # saved_model_path = "data/twitter_bert"
    # classifier_model.save(saved_model_path, include_optimizer=False)
    #
    data = clean_data(
        pd.read_pickle("data/df.pk"), fasttext_model_location="data/lid.176.bin"
    )
    data["avis"] = data.Avis.map(cat_codes.get)
    print(data.avis)
    data.Tweet = lemmatize(data.Tweet)

    train, val, test = train_val_test_split(data)

    print(predict(val.Tweet))

    from sklearn.metrics import accuracy_score

    val["pred"] = predict(val.Tweet)
    print(val.pred)
    val[val.Language != "en"].pred = 3
    print(accuracy_score(val.avis, val.pred))
