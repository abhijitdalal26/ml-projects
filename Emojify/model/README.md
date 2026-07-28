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

## How to generate them (Kaggle, recommended — clone and run, no pasting)

1. Create a new Kaggle Notebook.
2. Settings -> Add Input -> search `glove6b50dtxt` -> add that public dataset.
   `train.py` auto-detects it at `/kaggle/input/glove6b50dtxt/glove.6B.50d.txt`.
3. In a single code cell:
   ```
   !git clone https://github.com/abhijitdalal26/ml-projects.git
   %cd ml-projects/Emojify/train-model
   !pip install -q tensorflowjs
   !python train.py
   ```
4. `train.py` writes `model.json`, `group1-shard*.bin`, and `vocab.json`
   directly into the cloned repo's `Emojify/model/`, and also zips them to
   `/kaggle/working/emojify_model.zip` so they're easy to grab from the
   notebook's **Output** tab.
5. Download `emojify_model.zip`, unzip it, and drop the 3 files into this
   folder (`Emojify/model/`) in your local repo.
6. Reload `Emojify/index.html` — the demo will pick the files up
   automatically.

## Colab / local alternative

Same script, but GloVe needs fetching manually and there's no auto-zip step:
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove6b
```
Make sure `glove6b/glove.6B.50d.txt` is reachable from the script's working
directory (see `GLOVE_CANDIDATES` in `train.py`), then run `python train.py`
from inside `train-model/` — it writes straight into `../model/`. If GloVe
isn't found at all, the script falls back to random embeddings so it still
runs (accuracy will just be lower).

Once this is done, update the root `D:/ML-Projects/CLAUDE.md` to move Emojify
from "In progress" to "Completed".
