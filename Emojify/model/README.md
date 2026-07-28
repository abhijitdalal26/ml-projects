# Emojify model

This folder is where the trained/exported model files go after running
`../train-model/train.py`. It's currently empty — the web demo (`../index.html`)
will show a "model not trained yet" message until these files exist.

## What needs to be here

- `model.json` — TF.js model architecture + manifest
- `group1-shard1of1.bin` (and any additional shards) — model weights
- `vocab.json` — `{ word_to_idx, max_len, labels, emojis }`, used by `script.js`
  to tokenize/pad input sentences the same way the model was trained

All three are produced automatically by `train.py`.

## How to generate them

1. Open `train-model/train.py` on Google Colab or Kaggle (paste cell-by-cell,
   or upload and run `python train.py`).
2. Get GloVe embeddings (either works):
   - **Kaggle**: add the public dataset `glove6b50dtxt` to your notebook —
     `train.py` auto-detects it at `/kaggle/input/glove6b50dtxt/glove.6B.50d.txt`.
   - **Colab / local**: download and unzip manually:
     ```
     wget http://nlp.stanford.edu/data/glove.6B.zip
     unzip glove.6B.zip -d glove6b
     ```
     then make sure `glove6b/glove.6B.50d.txt` is reachable from the script's
     working directory (see `GLOVE_CANDIDATES` in `train.py`).
   - If GloVe isn't found at all, `train.py` falls back to random embeddings
     so the script still runs — accuracy will just be lower.
3. Run the script top to bottom. It trains an LSTM classifier over 5 emoji
   classes (love, baseball, smile, sad, food), evaluates it, and exports:
   - `model.json` + weight shard(s) via `tensorflowjs_converter`
     (`tensorflowjs.converters.save_keras_model`)
   - `vocab.json`
4. Copy/download `model.json`, the `group1-shard*.bin` file(s), and
   `vocab.json` into this folder (`Emojify/model/`).
5. Reload `Emojify/index.html` — the demo will pick the files up automatically.

Once this is done, update the root `D:/ML-Projects/CLAUDE.md` to move Emojify
from "In progress" to "Completed".
