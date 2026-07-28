Run `train-model/train.py` on Colab or Kaggle (plain Python script, not a
notebook — paste it into a cell or run it as a script). It trains on the
built-in Keras IMDB dataset and writes its output directly into this folder:

- `model.json` + `group1-shard*.bin` — the TensorFlow.js model
- `vocab.json` — the word -> index mapping used to tokenize input text in the browser

If the script is run somewhere else, just copy those files into this folder
afterward (same layout as `MNIST/model/`).
