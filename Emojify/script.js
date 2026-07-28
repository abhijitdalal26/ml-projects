// Emojify — client-side inference
//
// Loads a TF.js model + vocab.json (both produced by train-model/train.py),
// tokenizes/pads the input sentence the same way the training script did,
// runs the model, and shows the predicted emoji + confidence.

const MODEL_URL = "model/model.json";
const VOCAB_URL = "model/vocab.json";

const sentenceInput = document.getElementById("sentence-input");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const statusMessage = document.getElementById("status-message");
const predictionEmoji = document.getElementById("prediction-emoji");
const confidenceText = document.getElementById("confidence-text");

let model = null;
let vocab = null; // { word_to_idx, max_len, labels, emojis }

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[,.!?]/g, "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function sentenceToSequence(text) {
  const { word_to_idx, max_len } = vocab;
  const tokens = tokenize(text);
  const idxs = tokens
    .slice(0, max_len)
    .map((w) => (Object.prototype.hasOwnProperty.call(word_to_idx, w) ? word_to_idx[w] : 1));
  while (idxs.length < max_len) idxs.push(0); // pad with 0
  return idxs;
}

async function loadModel() {
  predictBtn.disabled = true;
  setStatus("Loading model...");
  try {
    const [loadedModel, vocabResponse] = await Promise.all([
      tf.loadLayersModel(MODEL_URL),
      fetch(VOCAB_URL),
    ]);

    if (!vocabResponse.ok) {
      throw new Error("vocab.json not found");
    }

    model = loadedModel;
    vocab = await vocabResponse.json();

    setStatus("Model ready. Type a sentence and click Emojify.");
    predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Model not trained yet — see model/README.md", true);
    predictBtn.disabled = true;
  }
}

function predict() {
  const text = sentenceInput.value.trim();
  if (!text) {
    setStatus("Type a sentence first.", true);
    return;
  }
  if (!model || !vocab) {
    setStatus("Model not trained yet — see model/README.md", true);
    return;
  }

  const seq = sentenceToSequence(text);
  const inputTensor = tf.tensor2d([seq], [1, vocab.max_len]);

  tf.tidy(() => {
    const output = model.predict(inputTensor);
    const probs = output.dataSync();
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const emoji = vocab.emojis[bestIdx] || "?";
    const confidence = probs[bestIdx] * 100;

    predictionEmoji.textContent = emoji;
    confidenceText.textContent = `${confidence.toFixed(1)}%`;
  });

  inputTensor.dispose();
  setStatus("Prediction complete.");
}

function clearAll() {
  sentenceInput.value = "";
  predictionEmoji.textContent = "🙂";
  confidenceText.textContent = "--";
  setStatus(model ? "Model ready. Type a sentence and click Emojify." : "");
  sentenceInput.focus();
}

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearAll);
sentenceInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    predict();
  }
});

loadModel();
