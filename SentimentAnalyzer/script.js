// Grab elements
const reviewInput = document.getElementById("review-input");
const analyzeBtn = document.getElementById("analyze-btn");
const statusMessage = document.getElementById("status-message");
const predictText = document.getElementById("prediction-text");
const confidenceFill = document.getElementById("confidence-bar-fill");
const confidenceValue = document.getElementById("confidence-value");

let model = null;
let vocab = null; // { word_index, vocab_size, max_len, pad_token, start_token, oov_token }

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
}

function setControlsEnabled(enabled) {
  analyzeBtn.disabled = !enabled;
}

// LOAD THE MODEL + VOCAB (async, in background)
async function init() {
  setControlsEnabled(false);
  setStatus("Loading model...");

  try {
    const [loadedModel, vocabResponse] = await Promise.all([
      tf.loadLayersModel("model/model.json"),
      fetch("model/vocab.json"),
    ]);

    if (!vocabResponse.ok) throw new Error("vocab.json not found");

    model = loadedModel;
    vocab = await vocabResponse.json();

    setStatus("Model loaded. Type a review and click Analyze.");
    setControlsEnabled(true);
  } catch (error) {
    console.error("Failed to load model:", error);
    setStatus(
      "Model not trained yet — see model/README.md",
      true
    );
    setControlsEnabled(false);
  }
}
init();

// TOKENIZE input text into a padded/truncated sequence of indices
function tokenize(text) {
  const { word_index, max_len, pad_token, start_token, oov_token } = vocab;

  // lowercase, strip punctuation, split on whitespace
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);

  // map words to indices with OOV fallback; prepend the start token, since
  // the model was trained on sequences produced by keras.datasets.imdb.load_data
  // (which prepends a start token to every sequence by default)
  const indices = [start_token];
  for (const word of words) {
    const idx = word_index[word];
    indices.push(idx !== undefined ? idx : oov_token);
  }

  // pad/truncate to max_len (post-padding/truncating, matching training)
  const sequence = new Array(max_len).fill(pad_token);
  for (let i = 0; i < Math.min(indices.length, max_len); i++) {
    sequence[i] = indices[i];
  }

  return sequence;
}

// RUN THE PREDICTION PIPELINE
function analyze() {
  if (!model || !vocab) return;

  const text = reviewInput.value.trim();
  if (!text) {
    setStatus("Type something first!", true);
    return;
  }

  setStatus("");

  tf.tidy(() => {
    const sequence = tokenize(text);
    const input = tf.tensor2d([sequence], [1, sequence.length]);

    const prediction = model.predict(input);
    const score = prediction.dataSync()[0]; // 0 (negative) .. 1 (positive)

    const isPositive = score >= 0.5;
    const confidencePct = Math.round((isPositive ? score : 1 - score) * 100);

    predictText.textContent = isPositive ? "Positive 😀" : "Negative 😞";
    confidenceFill.style.width = `${confidencePct}%`;
    confidenceValue.textContent = `${confidencePct}%`;
  });
}

analyzeBtn.addEventListener("click", analyze);

// Allow Ctrl/Cmd+Enter to trigger analysis from the textarea
reviewInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyze();
  }
});
