# Emojify — train.py
#
# Maps a short sentence to one of 5 emojis using GloVe word embeddings +
# an LSTM classifier. Designed to run top-to-bottom on Kaggle/Colab, either
# as `python train.py` or pasted cell-by-cell into a notebook.
#
# ---------------------------------------------------------------------------
# KAGGLE QUICKSTART — clone-and-run, no copy/paste
#   1. New Notebook -> Settings -> Add Input -> search "glove6b50dtxt" ->
#      add that dataset (public, by watts2 / thanakomsn).
#   2. In a code cell:
#        !git clone https://github.com/abhijitdalal26/ml-projects.git
#        %cd ml-projects/Emojify/train-model
#        !pip install -q tensorflowjs
#        !python train.py
#   3. This writes model.json, group1-shard*.bin, and vocab.json straight
#      into the cloned repo's Emojify/model/ — and also zips them to
#      /kaggle/working/emojify_model.zip so you can grab them from the
#      notebook's Output tab without digging through the clone.
#   4. Download emojify_model.zip, unzip it, drop the 3 files into
#      Emojify/model/ in your local repo, reload Emojify/index.html.
# ---------------------------------------------------------------------------

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Emoji classes
# ---------------------------------------------------------------------------
EMOJIS = ["❤️", "⚾", "😄", "😞", "🍴"]
LABEL_NAMES = ["heart", "baseball", "smile", "sad", "fork_and_knife"]
NUM_CLASSES = len(EMOJIS)

# ---------------------------------------------------------------------------
# 2. Small embedded dataset (~180 short labeled sentences, 5 classes)
#    label indices: 0=heart(love) 1=baseball(sport) 2=smile(happy)
#                   3=sad(disappointed) 4=fork_and_knife(food)
# ---------------------------------------------------------------------------
DATA = []

heart_sentences = [
    "I love you so much", "I adore you", "You mean the world to me",
    "My heart is full of love", "I love you with all my heart",
    "You are my sweetheart", "I cherish every moment with you",
    "Sending you all my love", "I am so in love", "You are so precious to me",
    "I love spending time with my family", "She is the love of my life",
    "I love this so much", "My love for you grows every day",
    "I will always love you", "You have my heart forever",
    "I love my best friend dearly", "Missing you and loving you always",
    "I love my mom and dad", "You make my heart happy",
    "I truly love this gift", "I love you to the moon and back",
    "I feel so much love right now", "My heart beats for you",
    "I love this beautiful day with you", "Nothing compares to my love for you",
    "I love my dog so much", "I love how caring you are",
    "You are adored by everyone", "I love being with you",
    "I love my grandparents so dearly", "I have so much love in my heart",
    "I love this romantic evening", "I love you more than words can say",
    "My heart melts when I see you", "I am grateful for your love",
]
DATA += [(s, 0) for s in heart_sentences]

baseball_sentences = [
    "Let's go play baseball", "He hit a home run", "The pitcher threw a strike",
    "I love watching baseball games", "She caught the fly ball",
    "We won the baseball championship", "He is a great baseball player",
    "The umpire called it a strike", "Let's play catch in the yard",
    "The batter swung and missed", "I want to watch the baseball game tonight",
    "He pitched a perfect game", "The team practiced batting today",
    "I threw the ball to first base", "He stole second base",
    "The crowd cheered for the home run", "Let's go to the baseball stadium",
    "She plays softball every weekend", "He is the best hitter on the team",
    "The baseball season starts soon", "We are playing baseball this afternoon",
    "The coach called a timeout", "He wore his baseball glove",
    "The ball flew over the fence", "I bought tickets to the baseball game",
    "The pitcher warmed up in the bullpen", "He slid into home plate",
    "Our team is playing baseball tonight", "She hit a triple",
    "The baseball bat cracked loudly", "Grab your glove, let's play ball",
    "The World Series game was exciting", "He plays shortstop for his team",
    "I enjoy going to baseball practice", "The umpire called him out",
    "We watched the baseball match together",
]
DATA += [(s, 1) for s in baseball_sentences]

smile_sentences = [
    "I am so happy today", "This is the best day ever", "I am laughing so hard",
    "That joke was hilarious", "I feel great right now", "What a wonderful surprise",
    "I am so excited for the party", "This makes me so happy",
    "We had so much fun today", "I can't stop smiling", "That was such a fun trip",
    "I am thrilled about the news", "This is amazing news", "I feel fantastic today",
    "We celebrated all night long", "I got the job, I am so happy",
    "This is hilarious, I am cracking up", "I am delighted to see you",
    "Today was an awesome day", "I am so glad you are here",
    "This party is so much fun", "I passed my exam, I am overjoyed",
    "We are having a blast", "I am beaming with joy",
    "This is the funniest thing ever", "I feel wonderful and cheerful",
    "That movie was so funny", "I am grinning from ear to ear",
    "I love how funny you are", "This surprise made my day",
    "I am jumping for joy", "We won the game, so exciting",
    "Life is good, I am happy", "I feel so cheerful this morning",
    "This is such a joyful moment",
]
DATA += [(s, 2) for s in smile_sentences]

sad_sentences = [
    "I am feeling really down today", "This makes me so sad",
    "I am disappointed in the results", "I feel so upset right now",
    "That was such a sad movie", "I am heartbroken over this",
    "I failed my exam and I feel awful", "This news makes me miserable",
    "I am crying right now", "I feel so lonely today",
    "That was a terrible day", "I am so unhappy with this outcome",
    "This is really depressing", "I miss you so much it hurts",
    "I feel like giving up", "Everything is going wrong today",
    "I am devastated by the news", "This breaks my heart",
    "I feel so gloomy today", "I am really discouraged right now",
    "That was such a disappointing game", "I feel empty inside",
    "I am so hurt by what happened", "This situation makes me sad",
    "I feel down in the dumps", "I am grieving the loss",
    "I feel so blue today", "This is such a sorrowful story",
    "I am upset that we lost", "I feel dejected after the news",
    "I am so sorry this happened", "I feel awful about the mistake",
    "This is a really gloomy day", "I am mourning this loss",
    "I feel defeated and sad",
]
DATA += [(s, 3) for s in sad_sentences]

food_sentences = [
    "Let's go get some food", "I am so hungry right now", "This pizza is delicious",
    "I want to eat dinner now", "Let's order some pasta", "I love eating pizza",
    "This restaurant has great food", "I am craving a burger",
    "Let's cook dinner together", "I want to try the new sushi place",
    "This cake tastes amazing", "I am starving, let's eat",
    "We are having pasta for dinner", "I love homemade cookies",
    "Let's grab lunch at the diner", "This soup is so tasty",
    "I want a sandwich for lunch", "Let's bake a cake today",
    "I am eating breakfast right now", "This steak is cooked perfectly",
    "I love spicy tacos", "Let's have a barbecue this weekend",
    "I want some ice cream for dessert", "This meal is absolutely delicious",
    "I am going to the buffet for dinner", "Let's order Chinese food tonight",
    "I love fresh baked bread", "This curry is so flavorful",
    "I want to make spaghetti tonight", "Let's eat at the new restaurant",
    "I am hungry for some noodles", "This chocolate cake is amazing",
    "I love cooking a big feast", "Let's have pancakes for breakfast",
    "I want a slice of pizza",
]
DATA += [(s, 4) for s in food_sentences]

sentences = [s for s, _ in DATA]
labels = np.array([l for _, l in DATA], dtype=np.int32)
print(f"Total examples: {len(sentences)}")

# ---------------------------------------------------------------------------
# 3. Build vocabulary
# ---------------------------------------------------------------------------
MAX_LEN = 10  # max words per sentence (pad/truncate)


def tokenize(text):
    return text.lower().replace(",", "").replace(".", "").replace("!", "").split()


word_counts = {}
for s in sentences:
    for w in tokenize(s):
        word_counts[w] = word_counts.get(w, 0) + 1

# index 0 = padding, index 1 = unknown/OOV
vocab = ["<pad>", "<unk>"] + sorted(word_counts.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
print(f"Vocab size: {VOCAB_SIZE}")


def sentence_to_seq(text):
    idxs = [word_to_idx.get(w, 1) for w in tokenize(text)]
    idxs = idxs[:MAX_LEN]
    idxs = idxs + [0] * (MAX_LEN - len(idxs))
    return idxs


X = np.array([sentence_to_seq(s) for s in sentences], dtype=np.int32)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Load GloVe embeddings (with a random-embedding fallback so this never
#    crashes if GloVe isn't available in the environment).
#
#    Common locations to try:
#      - Kaggle's built-in dataset: /kaggle/input/glove6b50dtxt/glove.6B.50d.txt
#        (add the "glove6b50dtxt" dataset to your Kaggle notebook)
#      - Local/Colab download:
#          !wget -q http://nlp.stanford.edu/data/glove.6B.zip
#          !unzip -q glove.6B.zip -d glove6b
#        then GLOVE_PATH = "glove6b/glove.6B.50d.txt"
# ---------------------------------------------------------------------------
EMBED_DIM = 50
GLOVE_CANDIDATES = [
    "/kaggle/input/glove6b50dtxt/glove.6B.50d.txt",
    "/kaggle/input/glove-6b-50d/glove.6B.50d.txt",
    "glove.6B.50d.txt",
    "glove6b/glove.6B.50d.txt",
]


def find_glove_path():
    for p in GLOVE_CANDIDATES:
        if os.path.exists(p):
            return p
    matches = glob.glob("/kaggle/input/**/glove.6B.50d.txt", recursive=True)
    if matches:
        return matches[0]
    return None


def load_glove(path, embed_dim, word_to_idx, vocab_size):
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embed_dim)).astype(
        "float32"
    )
    embedding_matrix[0] = np.zeros(embed_dim)  # <pad>
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word_to_idx:
                vec = np.asarray(parts[1:], dtype="float32")
                if vec.shape[0] == embed_dim:
                    embedding_matrix[word_to_idx[word]] = vec
                    found += 1
    print(f"GloVe: found embeddings for {found}/{vocab_size} vocab words")
    return embedding_matrix


glove_path = find_glove_path()
if glove_path:
    print(f"Loading GloVe from: {glove_path}")
    embedding_matrix = load_glove(glove_path, EMBED_DIM, word_to_idx, VOCAB_SIZE)
else:
    print("GloVe file not found — falling back to random embeddings.")
    embedding_matrix = np.random.uniform(
        -0.05, 0.05, (VOCAB_SIZE, EMBED_DIM)
    ).astype("float32")
    embedding_matrix[0] = np.zeros(EMBED_DIM)

# ---------------------------------------------------------------------------
# 5. Build model: Embedding (GloVe-initialized) -> LSTM -> Dense softmax
# ---------------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBED_DIM,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True,
            name="embedding",
        ),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=2,
)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

# Quick manual sanity check
sample_sentences = [
    "I love you",
    "Let's play baseball",
    "I am so happy",
    "I feel really sad",
    "Let's eat pizza",
]
sample_X = np.array([sentence_to_seq(s) for s in sample_sentences])
preds = model.predict(sample_X)
for s, p in zip(sample_sentences, preds):
    idx = int(np.argmax(p))
    print(f"{s!r:35s} -> {EMOJIS[idx]} ({LABEL_NAMES[idx]}, conf={p[idx]:.2f})")

# ---------------------------------------------------------------------------
# 8. Save vocab.json (word -> index) for the browser demo
# ---------------------------------------------------------------------------
import json

# Always write into the repo's own Emojify/model/ folder, relative to this
# script's location (train-model/). This works whether the repo was cloned
# into /kaggle/working, pulled in Colab, or run locally — no path surgery
# needed, just `cd Emojify/train-model && python train.py`.
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

vocab_export = {
    "word_to_idx": word_to_idx,
    "max_len": MAX_LEN,
    "labels": LABEL_NAMES,
    "emojis": EMOJIS,
}
with open(os.path.join(MODEL_DIR, "vocab.json"), "w", encoding="utf8") as f:
    json.dump(vocab_export, f)
print(f"Saved vocab.json to {MODEL_DIR}/vocab.json")

# Also save a Keras model file in case tfjs conversion needs to be re-run later
keras_model_path = os.path.join(MODEL_DIR, "emojify_keras_model.keras")
model.save(keras_model_path)
print(f"Saved Keras model to {keras_model_path}")

# ---------------------------------------------------------------------------
# 9. Export to TensorFlow.js format
#    Requires: pip install tensorflowjs
# ---------------------------------------------------------------------------
try:
    import tensorflowjs as tfjs

    tfjs.converters.save_keras_model(model, MODEL_DIR)
    print(f"Exported TF.js model to {MODEL_DIR}/ (model.json + weight shards)")

    # Keras 3 writes InputLayer config as "batch_shape", but the TF.js
    # layers loader in the browser still only understands the older
    # "batchInputShape" key — without this patch the browser throws
    # "An InputLayer should be passed either a batchInputShape or an
    # inputShape" and the model never loads.
    model_json_path = os.path.join(MODEL_DIR, "model.json")
    with open(model_json_path, encoding="utf8") as f:
        model_json = json.load(f)
    topology_config = model_json["modelTopology"].get(
        "model_config", model_json["modelTopology"]
    )["config"]
    patched = 0
    for layer in topology_config["layers"]:
        if layer.get("class_name") == "InputLayer" and "batch_shape" in layer["config"]:
            layer["config"]["batchInputShape"] = layer["config"].pop("batch_shape")
            patched += 1
    # Keras 3's converter also writes some weight names prefixed with the
    # outer model's own name (e.g. "sequential/dense/kernel") while others
    # (e.g. "embedding/embeddings") are left unprefixed. The browser loader
    # expects every weight name to be layer-relative, so an inconsistent
    # prefix causes "Provided weight data has no target variable: ...".
    # Strip it everywhere for consistency.
    model_name = topology_config.get("name", "sequential")
    prefix = model_name + "/"
    for group in model_json["weightsManifest"]:
        for weight in group["weights"]:
            if weight["name"].startswith(prefix):
                weight["name"] = weight["name"][len(prefix):]
                patched += 1

    if patched:
        with open(model_json_path, "w", encoding="utf8") as f:
            json.dump(model_json, f)
        print(f"Patched {patched} entries in model.json for TF.js compatibility")
except ImportError:
    print(
        "tensorflowjs not installed. Run:\n"
        "  pip install tensorflowjs\n"
        "then re-run just this conversion step:\n"
        "  import tensorflowjs as tfjs\n"
        "  tfjs.converters.save_keras_model(model, '../model')"
    )

# ---------------------------------------------------------------------------
# 10. On Kaggle, also zip model.json + weight shard(s) + vocab.json into
#     /kaggle/working/emojify_model.zip so they show up in the Output tab
#     as a single one-click download (the clone itself isn't listed there).
# ---------------------------------------------------------------------------
if os.path.isdir("/kaggle/working"):
    import shutil

    zip_files = [
        f
        for f in os.listdir(MODEL_DIR)
        if f == "model.json" or f == "vocab.json" or f.startswith("group1-shard")
    ]
    staging_dir = "/kaggle/working/emojify_model"
    os.makedirs(staging_dir, exist_ok=True)
    for f in zip_files:
        shutil.copy(os.path.join(MODEL_DIR, f), os.path.join(staging_dir, f))
    zip_path = shutil.make_archive("/kaggle/working/emojify_model", "zip", staging_dir)
    print(f"Zipped model files to {zip_path} — download from the Output tab.")
