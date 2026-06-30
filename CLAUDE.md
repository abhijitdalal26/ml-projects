# ML Projects — Codebase Guide

## Repo goal
Build ML/AI projects: train in Python (Jupyter notebook), then either deploy as a browser demo (TensorFlow.js / ONNX.js) or keep as a notebook. All browser demos live on GitHub Pages.

## Structure
```
/                  → landing page (index.html + style.css)
/MNIST/            → digit recognizer demo (deployed)
  train-model/     → Keras training notebook
  model/           → exported TF.js model weights
```

## Live site
https://abhijitdalal26.github.io/ml-projects/

---

## Completed
- **MNIST Digit Recognizer** — CNN trained in Keras, deployed with TensorFlow.js. Draw a digit, get a real-time prediction.

---

## Project ideas (future)

### Computer Vision
- **Fashion MNIST classifier** — same pipeline as MNIST but for clothing categories (shirt, shoe, bag…)
- **Face emotion detector** — train on FER-2013, deploy webcam feed in browser using face-api.js or TF.js
- **Object detector (YOLO/MobileNet)** — real-time webcam object detection in browser
- **Image style transfer** — apply artistic styles to uploaded photos (neural style transfer notebook)
- **Sketch-to-label classifier** — QuickDraw dataset, recognize doodles

### Natural Language Processing
- **Sentiment analyzer** — train on IMDB/Twitter data, deploy a text-input demo page
- **Spam classifier** — classic Naive Bayes / logistic regression notebook
- **Text summarizer** — extractive or abstractive summarization notebook (transformers)
- **Language detector** — identify the language of a typed sentence

### Classic ML (notebook-only is fine)
- **Titanic survival predictor** — classic Kaggle problem, full EDA + model comparison notebook
- **House price predictor** — regression with feature engineering, interactive price estimator
- **Customer churn predictor** — classification with imbalanced data techniques
- **K-Means image color quantizer** — reduce an image to N colors, show before/after in browser

### Generative / Fun
- **Digit generator (GAN/VAE)** — train a VAE on MNIST, let users generate digits in browser
- **Music genre classifier** — train on GTZAN audio features, upload an audio clip and get a genre
- **Rock Paper Scissors** — train on hand gesture images from webcam, play in browser

### Reinforcement Learning (notebook)
- **CartPole with DQN** — classic RL environment, visualize training curves
- **Tic-Tac-Toe agent** — self-play Q-learning, deploy a playable browser version

---

## Workflow for each project
1. Train / experiment in `project-name/train-model/` (Jupyter notebook)
2. Export model (`.h5` → TF.js, or `.onnx` for ONNX.js, or pickle for server-side)
3. Build browser demo in `/project-name/` or document as notebook-only
4. Add a card on the home `index.html`
5. Update this file — move the idea from "future" to "completed"
