# ML Projects

A collection of interactive machine learning demos that run entirely in the browser — no installs, no backend.

**Live site:** https://abhijitdalal26.github.io/ml-projects/

---

## Projects

### MNIST Digit Recognizer
Draw a digit (0–9) on the canvas and the model predicts it in real time.

- Built with TensorFlow.js — the CNN model runs locally in your browser
- Works on desktop (mouse) and mobile (touch)
- Shows the 28×28 model input alongside the prediction

---

## How it works

Each project is a self-contained folder with plain HTML, CSS, and JavaScript. The trained model weights are bundled with the code, so everything loads from static files — nothing is sent to a server.

---

## Tech stack

- **TensorFlow.js** — in-browser inference
- **Python / Keras** — used to train the original model (see `MNIST/train-model/`)
- **GitHub Pages** — hosting
