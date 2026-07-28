# ML Projects — Codebase Guide

## Repo goal
Build ML/AI projects: train in Python (Jupyter notebook), then either deploy as a browser demo (TensorFlow.js / ONNX.js) or keep as a notebook. All browser demos live on GitHub Pages.

## Structure
```
/                  → landing page (index.html + style.css)
/MNIST/            → digit recognizer demo (deployed)
  train-model/     → Keras training notebook
  model/           → exported TF.js model weights
/FashionMNIST/     → "Fashion Draw" sketch game (pipeline ready, awaiting trained model)
  train-model/     → Keras training notebook (run on Colab)
  model/           → exported TF.js model weights (drop in after running the notebook)
/SentimentAnalyzer/ → IMDB sentiment analyzer (pipeline ready, awaiting trained model)
  train-model/     → Keras training script (train.py, run on Colab/Kaggle)
  model/           → exported TF.js model weights + vocab.json (drop in after running train.py)
/ImageCaptioning/  → CNN (MobileNetV2) encoder + LSTM decoder image captioner (pipeline ready, awaiting trained model)
  train-model/     → Keras training script (train.py, run on Kaggle — flickr8k dataset built in)
  model/           → exported TF.js decoder + model/encoder/ (MobileNetV2) + tokenizer.json (drop in after running train.py)
/Emojify/          → sentence-to-emoji demo (deployed)
  train-model/     → train.py, plain Python script (run on Colab/Kaggle)
  model/           → exported TF.js model weights + vocab.json
```

## Live site
https://abhijitdalal26.github.io/ml-projects/

---

## Completed
- **MNIST Digit Recognizer** — CNN trained in Keras, deployed with TensorFlow.js. Draw a digit, get a real-time prediction.
- **Emojify** — type a short sentence, get the emoji that fits best (❤️ ⚾ 😄 😞 🍴). Model: Embedding (GloVe-initialized, 50d) → LSTM(64) → LSTM(32) → Dense softmax over 5 classes, trained on a small embedded dataset of ~180 hand-labeled sentences, 92.6% test accuracy. Web app (`Emojify/index.html`, `main.css`, `script.js`) tokenizes typed text client-side using `model/vocab.json` (lowercase, strip punctuation, split on whitespace, map to indices with OOV fallback, pad/truncate to `max_len`) and runs the TF.js model, showing the predicted emoji large plus a confidence score. Training script is `Emojify/train-model/train.py` (plain Python, run top-to-bottom on Kaggle — clone-and-run workflow documented at the top of the file and in `Emojify/model/README.md`; auto-loads GloVe from `/kaggle/input/glove6b50dtxt/`, falls back to random embeddings if not found). The script also self-patches two Keras 3 → TF.js browser-loader incompatibilities in the exported `model.json` after conversion: `InputLayer` configs use the new `batch_shape` key (TF.js's browser loader only understands `batchInputShape`), and the weight manifest inconsistently prefixes some weight names with the outer model's name (e.g. `sequential/dense/kernel`) while others aren't prefixed at all — both are normalized automatically so future retrains don't need manual fixing.

---

## In progress
- **Fashion Draw (Fashion MNIST game)** — Quick, Draw!-style game: app shows a random clothing category (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), player has 20s to sketch it on canvas, CNN predicts in real time as you draw, round ends on a confident correct guess or when the timer hits zero. Web app (`FashionMNIST/index.html`, `main.css`, `script.js`) and training notebook (`FashionMNIST/train-model/FashionMNIST-Notebook.ipynb`) are built. **Blocked on**: running the notebook on Colab (no GPU locally) and dropping the exported `model.json` + weight shard into `FashionMNIST/model/` — see `FashionMNIST/model/README.md`. Once the model is in place, move this entry to Completed.
- **Image Captioning** — CNN encoder (frozen MobileNetV2, GlobalAveragePooling) + LSTM decoder trained on Flickr8k; upload a photo, get a generated caption. Web app (`ImageCaptioning/index.html`, `main.css`, `script.js`) loads the MobileNetV2 encoder and the decoder as separate TF.js models and runs greedy decoding (feed `<start>`, repeatedly predict next word up to `max_length` or `<end>`) fully client-side. Training script `ImageCaptioning/train-model/train.py` (plain Python, run on Kaggle — `adityajn105/flickr8k` dataset is pre-attached at `/kaggle/input/flickr8k`) cleans/tokenizes captions, extracts image features, trains with teacher forcing, and exports the decoder + encoder to TF.js plus `tokenizer.json`. **Blocked on**: running `train.py` on Kaggle and dropping `model.json`+shards, `encoder/model.json`+shards, and `tokenizer.json` into `ImageCaptioning/model/` — see `ImageCaptioning/model/README.md`. Once the model is in place, move this entry to Completed.
- **Sentiment Analyzer** — type a movie review or any sentence, get real-time Positive/Negative classification. Model: Embedding → GlobalAveragePooling1D (or Bidirectional LSTM) → Dense → sigmoid, trained on the built-in Keras IMDB dataset (top 10k words). Web app (`SentimentAnalyzer/index.html`, `main.css`, `script.js`) tokenizes typed text client-side using `model/vocab.json` (lowercase, strip punctuation, split on whitespace, map to indices with OOV fallback, pad/truncate to the training sequence length) and runs the TF.js model, showing sentiment + a confidence bar. Training script is `SentimentAnalyzer/train-model/train.py` (plain Python, not a notebook — runs top-to-bottom on Colab/Kaggle). **Blocked on**: running `train.py` (no GPU locally, though not strictly required for this model) and dropping the exported `model.json` + weight shard + `vocab.json` into `SentimentAnalyzer/model/` — see `SentimentAnalyzer/model/README.md`. Note: if trained with Keras 3, the exported `model.json` may need the same TF.js compatibility patch documented in `Emojify/train-model/train.py` (InputLayer `batch_shape` → `batchInputShape`, and stripping inconsistent `sequential/` weight-name prefixes) before it will load in the browser. Once the model is in place, move this entry to Completed.

---

## Project ideas (future)

---

### Coursera — Machine Learning Specialization (Andrew Ng / MLS)
Projects from the MLS curriculum, rebuilt cleanly as standalone notebooks or demos.

- **Linear regression from scratch** — predict house prices; implement gradient descent manually, no sklearn
- **Logistic regression — tumor classifier** — binary classification (malignant/benign), decision boundary visualization
- **Neural network digit recognizer from scratch** — NumPy only, no framework; forward + backprop by hand
- **Decision tree & random forest** — predict heart disease; visualize tree splits
- **Recommender system** — collaborative filtering + content-based filtering on a movie dataset
- **Anomaly detection** — flag unusual server behavior using Gaussian distribution
- **K-Means clustering** — image compression: reduce photo to N colors; before/after browser demo
- **PCA visualization** — reduce high-dimensional data to 2D, interactive scatter plot

---

### Coursera — Deep Learning Specialization (Andrew Ng / DLS)
Classic DLS assignments, each turned into a clean notebook or deployed demo.

- **Cat vs Non-Cat classifier** — logistic regression then shallow NN, understand forward/backprop deeply
- **Deep NN from scratch** — build L-layer network in NumPy; classify images without any framework
- **Optimization algorithms notebook** — compare SGD, Momentum, RMSprop, Adam on a toy problem
- **Neural style transfer** — combine content + style images using VGG19; upload any two images
- **Face recognition** — FaceNet / Siamese network; verify "is this the same person?" from webcam
- **Trigger word detection** — audio spectrogram → RNN; detect a keyword in a recorded clip
- **Dinosaur name generator** — character-level RNN trained on dinosaur names; generates new names
- **Jazz music generator** — LSTM trained on jazz MIDI; generate a short jazz solo (notebook)
- **Machine translation** — attention-based seq2seq; translate short English phrases (notebook)
- **Named entity recognition (NER)** — BiLSTM tags names, places, orgs in a sentence
- **Trigger word detection with spectrograms** — visualize audio as spectrogram, run RNN inference

---

### Stanford CS229 — Machine Learning
- **Linear & polynomial regression** — Portland housing dataset; normal equation vs gradient descent
- **Logistic regression + regularization** — admission predictor with decision boundary plot
- **SVM with kernels** — spam classifier using an RBF kernel
- **K-Means + PCA** — compress and reconstruct an image; eigenfaces on face dataset
- **Anomaly detection** — network intrusion detection notebook
- **EM algorithm** — Gaussian Mixture Models, visualize clusters forming

### Stanford CS231n — Computer Vision
- **Image classifier (CIFAR-10)** — train from scratch; compare kNN → SVM → softmax → CNN
- **Backprop from scratch** — implement a modular neural net with NumPy (layers, activations, loss)
- **Batch norm & dropout notebook** — show effect on training curves
- **CNN feature visualization** — saliency maps, class activation maps (CAM) on any image
- **Object detection** — fine-tune YOLO or Faster R-CNN on a custom dataset
- **GAN — image generation** — DCGAN on MNIST or CelebA; generate fake faces or digits
- **U-Net image segmentation** — segment objects in an image pixel by pixel

### Stanford CS224n — NLP with Deep Learning
- **Word2Vec from scratch** — train skip-gram on a text corpus; visualize embeddings with t-SNE
- **Dependency parser** — transition-based parser notebook
- **Sentiment analysis with RNN/LSTM** — IMDB reviews; compare simple RNN vs LSTM vs GRU
- **Neural machine translation with attention** — English ↔ French seq2seq with attention heatmap
- **Transformer from scratch** — implement multi-head attention and positional encoding in NumPy/PyTorch
- **Question answering (SQuAD)** — fine-tune BERT on SQuAD; answer questions about a paragraph

### Stanford CS234 / MIT 6.S191 — Reinforcement Learning & Deep Learning
- **CartPole with DQN** — train a DQN agent; animate the pole balancing in browser (canvas)
- **Tic-Tac-Toe agent** — self-play Q-learning; deploy a playable browser version
- **Lunar Lander** — PPO or DQN on OpenAI Gym; visualize episode rewards over training
- **Pong from pixels** — policy gradient on raw pixel input (classic Karpathy blog post)

### MIT 6.S191 — Intro to Deep Learning (lab projects)
- **Music generation with RNN** — LSTM trained on MIDI; generate a melody (notebook + audio output)
- **Facial recognition & de-biasing** — build a face classifier, then audit and reduce demographic bias
- **Autonomous driving (lane following)** — CNN predicts steering angle from dashcam images
- **Reinforcement learning — Pong** — train an agent with policy gradients directly on pixel frames

---

### Explainability & Interpretability (XAI)
A whole area that's missing from most course lists but essential in real ML work.

- **SHAP values** — explain any model's predictions on tabular data; show which features matter most
- **LIME** — locally approximate any black-box model; explain a single image or text prediction
- **Grad-CAM** — highlight which pixels a CNN looked at when making a decision; overlay heatmap on image
- **Integrated Gradients** — attribution method for deep nets; visualize input importance
- **Partial dependence plots** — show how one feature affects predictions across its range
- **Confusion matrix deep-dive notebook** — per-class error analysis, worst-case sample inspection

---

### Time Series
- **Stock price prediction (LSTM)** — predict next-day price; show prediction vs actual chart
- **ARIMA vs LSTM comparison** — classical vs deep learning on the same time series dataset
- **Anomaly detection in time series** — flag unusual spikes in sensor/server data
- **Weather forecasting** — multi-step prediction using stacked LSTMs or temporal CNNs
- **ECG heartbeat classifier** — classify normal vs abnormal heartbeat from waveform data

---

### Modern Generative AI
- **Diffusion model from scratch** — implement DDPM (denoising diffusion) on MNIST; watch images denoise
- **Variational Autoencoder (VAE) — latent space explorer** — interpolate between two images in latent space
- **Conditional GAN** — generate digits/faces conditioned on a class label
- **Text-to-image with CLIP** — use OpenAI CLIP embeddings to search images by text description
- **Stable Diffusion fine-tuning (DreamBooth)** — fine-tune on a few photos of a subject (notebook)

---

### Transformers & Modern Architectures
- **Vision Transformer (ViT) from scratch** — patch-based image classification; compare to CNN
- **BERT fine-tuning** — sentiment / topic classification with a pre-trained transformer
- **GPT-2 fine-tuning** — fine-tune on a small custom text corpus; generate new text
- **CLIP zero-shot classifier** — classify images using only text descriptions, no labeled training data
- **DETR — object detection with transformers** — end-to-end detection without anchor boxes

---

### LLMs & Prompt Engineering
- **RAG (Retrieval-Augmented Generation)** — build a Q&A bot over your own documents using embeddings + LLM
- **LLM fine-tuning with LoRA** — efficiently fine-tune a small LLM (e.g. Mistral 7B) on custom data
- **Prompt engineering notebook** — systematically compare zero-shot, few-shot, chain-of-thought prompting
- **Embedding similarity search** — encode sentences, find nearest neighbors, build a semantic search demo
- **AI agent from scratch** — build a simple tool-using agent with function calling (notebook)

---

### Graph Neural Networks (GNNs)
- **Node classification** — classify nodes in a citation network (Cora dataset) with GCN
- **Link prediction** — predict missing edges in a social graph
- **Molecular property prediction** — predict if a molecule is toxic using graph structure (notebook)

---

### Speech & Audio
- **Speech command recognizer** — classify spoken words (yes/no/stop/go) from audio clips; deploy in browser (Web Audio API)
- **Speaker identification** — verify "is this the same speaker?" using mel-spectrograms + CNN
- **Speech emotion recognition** — classify emotion from voice (angry/happy/sad)

---

### Medical & Science Applications
- **Chest X-ray classifier** — detect pneumonia from X-rays (Kaggle dataset); discuss medical AI ethics
- **Brain tumor segmentation** — U-Net on MRI scans; pixel-level tumor detection (notebook)
- **Drug discovery notebook** — predict molecular activity using fingerprints + ML
- **COVID-19 trend forecasting** — time series prediction on real case data

---

### Bayesian & Probabilistic ML
- **Bayesian linear regression** — uncertainty-aware predictions; show confidence intervals
- **Gaussian Process regression** — smooth interpolation with uncertainty bands; interactive plot
- **Monte Carlo dropout** — turn dropout into uncertainty estimation at inference time
- **Naive Bayes from scratch** — spam classifier built manually; derive the math in the notebook

---

### Self-Supervised & Contrastive Learning
- **SimCLR** — learn image representations without labels using contrastive loss
- **Masked Autoencoder (MAE)** — reconstruct masked image patches; pre-train a ViT
- **Word2Vec contrastive** — understand skip-gram as a contrastive objective

---

### Model Optimization & Deployment
- **Quantization notebook** — shrink a trained model from float32 to int8; measure accuracy vs size tradeoff
- **Knowledge distillation** — train a small student network to mimic a large teacher
- **ONNX export pipeline** — export any Keras/PyTorch model to ONNX, run in browser via onnxruntime-web
- **Model drift detection** — monitor how predictions shift over time on new data (notebook)

---

### Fun / Portfolio Showcases
- **Neural network playground** — interactive visualizer (like TF Playground) built from scratch
- **Pose estimation** — use MoveNet/PoseNet in browser to track body keypoints from webcam
- **Real-time background blur** — segment person from background using MediaPipe; browser demo
- **License plate reader** — detect and OCR license plates from uploaded photos
- **Deepfake detector** — binary classifier to flag AI-generated face images

---

### Computer Vision (original ideas)
- **Fashion MNIST classifier** — same pipeline as MNIST but for clothing categories (shirt, shoe, bag…)
- **Face emotion detector** — train on FER-2013, deploy webcam feed in browser using face-api.js or TF.js
- **Object detector (YOLO/MobileNet)** — real-time webcam object detection in browser
- **Sketch-to-label classifier** — QuickDraw dataset, recognize doodles

### Natural Language Processing (original ideas)
- **Spam classifier** — classic Naive Bayes / logistic regression notebook
- **Text summarizer** — extractive or abstractive summarization notebook (transformers)
- **Language detector** — identify the language of a typed sentence

### Generative / Fun (original ideas)
- **Digit generator (GAN/VAE)** — train a VAE on MNIST, let users generate digits in browser
- **Music genre classifier** — train on GTZAN audio features, upload an audio clip and get a genre
- **Rock Paper Scissors** — train on hand gesture images from webcam, play in browser

---

## Workflow for each project
1. Train / experiment in `project-name/train-model/` (Jupyter notebook)
2. Export model (`.h5` → TF.js, or `.onnx` for ONNX.js, or pickle for server-side)
3. Build browser demo in `/project-name/` or document as notebook-only
4. Add a card on the home `index.html`
5. Update this file — move the idea from "future" to "completed"
