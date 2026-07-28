# Image Captioning — CNN encoder (MobileNetV2) + LSTM decoder
# Trained on the Flickr8k dataset.
#
# Designed to run top-to-bottom on Kaggle (dataset pre-attached at
# /kaggle/input/flickr8k) or Colab (see DATA_DIR override below).
#
# ---------------------------------------------------------------------------
# pip installs (Kaggle/Colab already ship most of these; uncomment if needed)
# ---------------------------------------------------------------------------
# !pip install tensorflow tensorflowjs pillow tqdm numpy
# ---------------------------------------------------------------------------

import json
import os
import re
import string

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Dense,
    Dropout,
    Embedding,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Kaggle dataset path (adityajn105/flickr8k). Override DATA_DIR for Colab,
# e.g. after unzipping into /content/flickr8k.
DATA_DIR = "/kaggle/input/flickr8k"
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")

OUTPUT_DIR = "/kaggle/working"
MODEL_TFJS_DIR = os.path.join(OUTPUT_DIR, "model_tfjs")           # decoder -> TF.js
ENCODER_TFJS_DIR = os.path.join(OUTPUT_DIR, "encoder_tfjs")       # MobileNetV2 -> TF.js
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "tokenizer.json")
KERAS_MODEL_PATH = os.path.join(OUTPUT_DIR, "decoder.keras")

IMG_SIZE = 224              # MobileNetV2 default input
FEATURE_DIM = 1280          # MobileNetV2 GlobalAveragePooling output dim
EMBED_DIM = 256
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 20
MAX_VOCAB_SIZE = 8000        # cap vocab so the exported tokenizer stays small
START_TOKEN = "startseq"
END_TOKEN = "endseq"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & clean captions
# ---------------------------------------------------------------------------


def load_captions(captions_file):
    """Parse captions.txt (format: image,caption) into {image_name: [captions]}."""
    mapping = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        next(f)  # skip header line: "image,caption"
        for line in f:
            line = line.strip()
            if not line:
                continue
            # captions may themselves contain commas, so split on the first one
            image_id, caption = line.split(",", 1)
            image_id = image_id.strip()
            mapping.setdefault(image_id, []).append(caption)
    return mapping


def clean_caption(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    # drop single-character "words" (noise from stray punctuation/digits)
    text = " ".join(w for w in text.split() if len(w) > 1 or w in ("a", "i"))
    return text


print("Loading captions...")
raw_mapping = load_captions(CAPTIONS_FILE)

captions_mapping = {}
for image_id, caps in raw_mapping.items():
    cleaned = [f"{START_TOKEN} {clean_caption(c)} {END_TOKEN}" for c in caps]
    captions_mapping[image_id] = cleaned

all_captions = [c for caps in captions_mapping.values() for c in caps]
print(f"Images: {len(captions_mapping)} | Captions: {len(all_captions)}")

# ---------------------------------------------------------------------------
# 2. Build vocabulary / tokenizer
# ---------------------------------------------------------------------------

print("Fitting tokenizer...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
max_length = max(len(c.split()) for c in all_captions)
print(f"Vocab size: {vocab_size} | Max caption length: {max_length}")

# Save tokenizer as plain JSON (word_index + index_word), consumed directly
# by the browser demo — no keras-specific deserialization needed client-side.
word_index = {w: i for w, i in tokenizer.word_index.items() if i < vocab_size}
index_word = {str(i): w for w, i in word_index.items()}

with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "word_index": word_index,
            "index_word": index_word,
            "vocab_size": vocab_size,
            "max_length": max_length,
            "start_token": START_TOKEN,
            "end_token": END_TOKEN,
        },
        f,
    )
print(f"Saved tokenizer -> {TOKENIZER_PATH}")

# ---------------------------------------------------------------------------
# 3. Extract image features with a frozen MobileNetV2
# ---------------------------------------------------------------------------

print("Building feature extractor (MobileNetV2, frozen, GAP)...")
base_cnn = MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
base_cnn.trainable = False
feature_extractor = Model(inputs=base_cnn.input, outputs=base_cnn.output)


def extract_features(image_ids, images_dir, batch_size=64):
    """Run MobileNetV2 over every image once and cache the feature vectors."""
    features = {}
    batch_imgs, batch_ids = [], []

    def flush():
        if not batch_imgs:
            return
        arr = np.stack(batch_imgs, axis=0)
        arr = preprocess_input(arr)
        preds = feature_extractor.predict(arr, verbose=0)
        for img_id, vec in zip(batch_ids, preds):
            features[img_id] = vec
        batch_imgs.clear()
        batch_ids.clear()

    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, image_id)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        batch_imgs.append(img_to_array(img))
        batch_ids.append(image_id)
        if len(batch_imgs) >= batch_size:
            flush()
        if (i + 1) % 500 == 0:
            print(f"  extracted features for {i + 1}/{len(image_ids)} images")
    flush()
    return features


print("Extracting image features (this is the slow step)...")
image_ids = list(captions_mapping.keys())
image_features = extract_features(image_ids, IMAGES_DIR)
print(f"Extracted features for {len(image_features)} images")

# ---------------------------------------------------------------------------
# 4. Train / validation split
# ---------------------------------------------------------------------------

np.random.seed(42)
shuffled_ids = image_ids.copy()
np.random.shuffle(shuffled_ids)
split_idx = int(0.9 * len(shuffled_ids))
train_ids = set(shuffled_ids[:split_idx])
val_ids = set(shuffled_ids[split_idx:])

# ---------------------------------------------------------------------------
# 5. Data generator (image feature, partial caption) -> next word
# ---------------------------------------------------------------------------


def data_generator(ids, captions_mapping, image_features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for image_id in ids:
            feature = image_features[image_id]
            for caption in captions_mapping[image_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                seq = [t for t in seq if t < vocab_size]
                for t in range(1, len(seq)):
                    in_seq, out_seq = seq[:t], seq[t]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []
                        n = 0


steps_per_epoch_train = max(
    1,
    sum(
        len(tokenizer.texts_to_sequences([c])[0]) - 1
        for i in train_ids
        for c in captions_mapping[i]
    )
    // BATCH_SIZE,
)
steps_per_epoch_val = max(
    1,
    sum(
        len(tokenizer.texts_to_sequences([c])[0]) - 1
        for i in val_ids
        for c in captions_mapping[i]
    )
    // BATCH_SIZE,
)

train_gen = data_generator(
    train_ids, captions_mapping, image_features, tokenizer, max_length, vocab_size, BATCH_SIZE
)
val_gen = data_generator(
    val_ids, captions_mapping, image_features, tokenizer, max_length, vocab_size, BATCH_SIZE
)

# ---------------------------------------------------------------------------
# 6. Build the encoder-decoder captioning model
# ---------------------------------------------------------------------------

print("Building captioning model...")

# Image feature branch (already-extracted MobileNetV2 GAP vector as input)
image_input = Input(shape=(FEATURE_DIM,), name="image_features")
image_dense = Dropout(0.5)(image_input)
image_dense = Dense(EMBED_DIM, activation="relu")(image_dense)

# Caption sequence branch
caption_input = Input(shape=(max_length,), name="caption_seq")
caption_embed = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(caption_input)
caption_embed = Dropout(0.5)(caption_embed)
caption_lstm = LSTM(LSTM_UNITS)(caption_embed)

# Merge image features with LSTM output, then predict next word
decoder = Add()([image_dense, caption_lstm])
decoder = Dense(LSTM_UNITS, activation="relu")(decoder)
output = Dense(vocab_size, activation="softmax")(decoder)

caption_model = Model(inputs=[image_input, caption_input], outputs=output)
caption_model.compile(loss="categorical_crossentropy", optimizer="adam")
caption_model.summary()

# ---------------------------------------------------------------------------
# 7. Train with teacher forcing
# ---------------------------------------------------------------------------

print("Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
]
caption_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch_train,
    validation_data=val_gen,
    validation_steps=steps_per_epoch_val,
    epochs=EPOCHS,
    callbacks=callbacks,
)

caption_model.save(KERAS_MODEL_PATH)
print(f"Saved trained decoder -> {KERAS_MODEL_PATH}")

# ---------------------------------------------------------------------------
# 8. Export decoder + encoder to TensorFlow.js
# ---------------------------------------------------------------------------

print("Exporting models to TensorFlow.js format...")
os.makedirs(MODEL_TFJS_DIR, exist_ok=True)
os.makedirs(ENCODER_TFJS_DIR, exist_ok=True)

# Decoder: image-features + caption-seq -> next-word softmax
os.system(
    f"tensorflowjs_converter --input_format keras "
    f"{KERAS_MODEL_PATH} {MODEL_TFJS_DIR}"
)

# Encoder: MobileNetV2 (frozen) -> 1280-d feature vector, so the browser can
# run feature extraction client-side with no backend.
encoder_keras_path = os.path.join(OUTPUT_DIR, "encoder.keras")
feature_extractor.save(encoder_keras_path)
os.system(
    f"tensorflowjs_converter --input_format keras "
    f"{encoder_keras_path} {ENCODER_TFJS_DIR}"
)

print("Done. Copy the following into ImageCaptioning/model/:")
print(f"  {MODEL_TFJS_DIR}/*        -> ImageCaptioning/model/ (model.json + shards)")
print(f"  {ENCODER_TFJS_DIR}/*      -> ImageCaptioning/model/encoder/ (model.json + shards)")
print(f"  {TOKENIZER_PATH}          -> ImageCaptioning/model/tokenizer.json")
print(f"max_length = {max_length}  <-- hardcode this constant into script.js")
