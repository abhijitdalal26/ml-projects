# Sentiment Analyzer — IMDB training script
#
# Designed to run top-to-bottom on Google Colab or Kaggle (with or without a GPU).
# Uses Keras's built-in IMDB dataset — no external download needed.
#
# --- Setup (run once, e.g. in a Colab cell) --------------------------------
# !pip install tensorflow tensorflowjs
# -----------------------------------------------------------------------------
#
# Output (written to ../model/):
#   model.json + group1-shard*.bin   <- TensorFlow.js model
#   vocab.json                       <- word -> index mapping used at inference time
#
# After running this script, drop the contents of ../model/ into the web app
# folder (they should already be there since OUTPUT_DIR points there directly).

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
VOCAB_SIZE = 10000       # top N most frequent words to keep
MAX_LEN = 200            # fixed sequence length (pad/truncate to this)
EMBEDDING_DIM = 16
BATCH_SIZE = 512
EPOCHS = 10

# Special token indices (must match what script.js expects)
PAD_TOKEN = 0
START_TOKEN = 1
OOV_TOKEN = 2
INDEX_FROM = 3  # real words are offset by this many slots for the special tokens above

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "model"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load the IMDB dataset (already integer-encoded by Keras)
# -----------------------------------------------------------------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE, index_from=INDEX_FROM
)
print(f"Train examples: {len(x_train)}, Test examples: {len(x_test)}")

# -----------------------------------------------------------------------------
# 2. Build word_index / index_word mappings
# -----------------------------------------------------------------------------
# keras.datasets.imdb.get_word_index() returns {word: index} with indices
# starting at 1 (no special tokens baked in). Since we loaded the dataset with
# index_from=INDEX_FROM, real word indices in x_train/x_test are shifted by
# INDEX_FROM, so we apply the same shift here to build a consistent mapping.
raw_word_index = keras.datasets.imdb.get_word_index()

word_index = {"<PAD>": PAD_TOKEN, "<START>": START_TOKEN, "<OOV>": OOV_TOKEN}
for word, idx in raw_word_index.items():
    new_idx = idx + INDEX_FROM
    if new_idx < VOCAB_SIZE:
        word_index[word] = new_idx

index_word = {v: k for k, v in word_index.items()}

# -----------------------------------------------------------------------------
# 3. Pad / truncate sequences to a fixed length
# -----------------------------------------------------------------------------
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_TOKEN
)
x_test = keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_TOKEN
)

# -----------------------------------------------------------------------------
# 4. Build the model
#    Embedding -> GlobalAveragePooling1D -> Dense -> sigmoid
#    (fast to train; swap GlobalAveragePooling1D for a Bidirectional LSTM
#     if you have GPU time and want to squeeze out a bit more accuracy)
# -----------------------------------------------------------------------------
USE_LSTM = False  # set True to use a Bidirectional LSTM instead of pooling

model = keras.Sequential()
model.add(layers.Input(shape=(MAX_LEN,)))
model.add(layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))

if USE_LSTM:
    model.add(layers.Bidirectional(layers.LSTM(32)))
else:
    model.add(layers.GlobalAveragePooling1D())

model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------------------------------------------------------
# 5. Train
# -----------------------------------------------------------------------------
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train_partial = x_train[5000:]
y_train_partial = y_train[5000:]

history = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2,
)

# -----------------------------------------------------------------------------
# 6. Evaluate
# -----------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# -----------------------------------------------------------------------------
# 7. Export vocab.json (word -> index, capped to VOCAB_SIZE, plus special tokens)
# -----------------------------------------------------------------------------
vocab_export = {
    "word_index": word_index,
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "pad_token": PAD_TOKEN,
    "start_token": START_TOKEN,
    "oov_token": OOV_TOKEN,
}

vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_export, f)
print(f"Saved vocab to {vocab_path}")

# -----------------------------------------------------------------------------
# 8. Export the trained model to TensorFlow.js format
# -----------------------------------------------------------------------------
# Save a Keras SavedModel first, then convert with the tensorflowjs_converter
# Python API (equivalent to running the CLI tool, but keeps everything in one
# script so it runs top-to-bottom without extra shell steps).
saved_model_dir = os.path.join(SCRIPT_DIR, "_saved_model")
model.export(saved_model_dir)

import tensorflowjs as tfjs

tfjs.converters.convert_tf_saved_model(saved_model_dir, OUTPUT_DIR)
print(f"Saved TensorFlow.js model to {OUTPUT_DIR}")

# Alternative one-liner if you prefer converting straight from the Keras model
# object without going through a SavedModel (uncomment if convert_tf_saved_model
# above gives you trouble in your environment):
# tfjs.converters.save_keras_model(model, OUTPUT_DIR)

print("Done! Copy model.json, the weight shard(s), and vocab.json into "
      "SentimentAnalyzer/model/ (already the OUTPUT_DIR above if run in place).")
