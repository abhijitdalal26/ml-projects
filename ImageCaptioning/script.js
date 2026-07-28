// Image Captioning demo — MobileNetV2 encoder (TF.js) + LSTM decoder (TF.js).
// All inference runs client-side; no backend required.

const MODEL_DIR = "model";
const ENCODER_URL = `${MODEL_DIR}/encoder/model.json`;
const DECODER_URL = `${MODEL_DIR}/model.json`;
const TOKENIZER_URL = `${MODEL_DIR}/tokenizer.json`;

// Hardcode this from the value train.py prints at the end of training
// (also stored in tokenizer.json as "max_length", used as a fallback).
const MAX_LENGTH = 34;
const IMG_SIZE = 224;

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const dropText = document.getElementById("drop-text");
const clearBtn = document.getElementById("clear-btn");
const captionText = document.getElementById("caption-text");

let encoder = null;
let decoder = null;
let tokenizer = null;
let modelsReady = false;

setCaption("Loading model...", "loading");
loadModels();

async function loadModels() {
  try {
    const [encoderModel, decoderModel, tokenizerJson] = await Promise.all([
      loadEncoder(),
      loadDecoder(),
      fetch(TOKENIZER_URL).then((r) => {
        if (!r.ok) throw new Error("missing tokenizer.json");
        return r.json();
      }),
    ]);
    encoder = encoderModel;
    decoder = decoderModel;
    tokenizer = tokenizerJson;
    modelsReady = true;
    setCaption("Upload an image to get started", "");
  } catch (err) {
    console.error(err);
    setCaption("Model not trained yet — see model/README.md", "error");
  }
}

async function loadEncoder() {
  try {
    return await tf.loadGraphModel(ENCODER_URL);
  } catch (e) {
    return await tf.loadLayersModel(ENCODER_URL);
  }
}

async function loadDecoder() {
  try {
    return await tf.loadGraphModel(DECODER_URL);
  } catch (e) {
    return await tf.loadLayersModel(DECODER_URL);
  }
}

// --- UI wiring -------------------------------------------------------------

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) handleFile(file);
});

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  previewImg.src = "";
  previewImg.hidden = true;
  dropText.hidden = false;
  setCaption("Upload an image to get started", "");
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropText.hidden = true;
    previewImg.onload = () => generateCaption(previewImg);
  };
  reader.readAsDataURL(file);
}

// --- Inference ---------------------------------------------------------

async function generateCaption(imgEl) {
  if (!modelsReady) {
    setCaption("Model not trained yet — see model/README.md", "error");
    return;
  }

  setCaption("Generating caption...", "loading");

  try {
    const featureTensor = tf.tidy(() => extractFeatures(imgEl));
    const caption = await decodeCaption(featureTensor);
    featureTensor.dispose();
    setCaption(caption, "");
  } catch (err) {
    console.error(err);
    setCaption("Something went wrong generating the caption.", "error");
  }
}

// MobileNetV2 preprocessing: resize to 224x224, scale pixel values to [-1, 1].
function extractFeatures(imgEl) {
  let input = tf.browser.fromPixels(imgEl).toFloat();
  input = tf.image.resizeBilinear(input, [IMG_SIZE, IMG_SIZE]);
  input = input.div(127.5).sub(1);
  input = input.expandDims(0);
  const output = encoder.predict(input);
  return Array.isArray(output) ? output[0] : output;
}

async function decodeCaption(featureTensor) {
  const { word_index: wordIndex, index_word: indexWord, start_token: startToken, end_token: endToken } =
    tokenizer;
  const maxLength = tokenizer.max_length || MAX_LENGTH;

  let sequence = [wordIndex[startToken]];
  const words = [];

  for (let step = 0; step < maxLength; step++) {
    const nextId = tf.tidy(() => {
      const padded = padSequence(sequence, maxLength);
      const seqTensor = tf.tensor2d([padded], [1, maxLength], "int32");
      const feats = featureTensor.reshape([1, featureTensor.shape[featureTensor.shape.length - 1]]);
      const preds = decoder.predict([feats, seqTensor]);
      const predsFlat = Array.isArray(preds) ? preds[0] : preds;
      return predsFlat.argMax(-1).dataSync()[0];
    });

    const word = indexWord[String(nextId)];
    if (!word || word === endToken) break;
    words.push(word);
    sequence.push(nextId);
    if (sequence.length >= maxLength) break;
  }

  return words.length ? words.join(" ") : "Couldn't come up with a caption for this one.";
}

// Left-pad/truncate a sequence of token ids to a fixed length (Keras
// pad_sequences default: pre-padding with zeros).
function padSequence(seq, maxLength) {
  const trimmed = seq.length > maxLength ? seq.slice(seq.length - maxLength) : seq;
  const padding = new Array(maxLength - trimmed.length).fill(0);
  return padding.concat(trimmed);
}

// --- Helpers ----------------------------------------------------------

function setCaption(text, mode) {
  captionText.textContent = text;
  captionText.classList.remove("loading", "error");
  if (mode) captionText.classList.add(mode);
}
