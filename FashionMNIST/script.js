// Fashion MNIST class order — must match the label order the model was trained on.
const CLASS_NAMES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

const ROUND_SECONDS = 20;
const CONFIDENCE_THRESHOLD = 0.6;

// Elements
const canvas = document.getElementById("main-canvas");
const ctx = canvas.getContext("2d");
const miniCanvas = document.getElementById("mini-canvas");
const miniCtx = miniCanvas.getContext("2d");

const promptText = document.getElementById("prompt-text");
const scoreText = document.getElementById("score-text");
const timerText = document.getElementById("timer-text");
const timerBar = document.getElementById("timer-bar");

const predictText = document.getElementById("prediction-text");
const confidenceText = document.getElementById("confidence-text");

const resultOverlay = document.getElementById("result-overlay");
const resultText = document.getElementById("result-text");

const clearBtn = document.getElementById("clear-btn");
const skipBtn = document.getElementById("skip-btn");
const startBtn = document.getElementById("start-btn");

// State
let model;
let score = 0;
let targetClass = null;
let roundActive = false;
let timeLeft = ROUND_SECONDS;
let timerInterval = null;
let roundEnding = false;

// LOAD THE MODEL (async, in the background)
async function loadModel() {
  try {
    console.log("Loading model...");
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded successfully!");
    startBtn.disabled = false;
  } catch (error) {
    console.error("Model failed to load:", error);
    predictText.innerText = "Error";
    promptText.innerText = "Model failed to load";
  }
}
startBtn.disabled = true;
loadModel();

// DRAWING SETTINGS
let isDrawing = false;
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
resetCanvas();

function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  miniCtx.fillStyle = "black";
  miniCtx.fillRect(0, 0, 28, 28);
}

const getPos = (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  return { x, y };
};

const startDrawing = (e) => {
  if (!roundActive) return;
  isDrawing = true;
  draw(e);
};
const stopDrawing = () => {
  isDrawing = false;
  ctx.beginPath();
};

const draw = (e) => {
  if (!isDrawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

  predictRealTime();
};

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);

canvas.addEventListener("touchstart", (e) => { e.preventDefault(); startDrawing(e.touches[0]); }, { passive: false });
canvas.addEventListener("touchmove", (e) => { e.preventDefault(); draw(e.touches[0]); }, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

// PREDICTION PIPELINE
function predictRealTime() {
  if (!model) return;

  tf.tidy(() => {
    let img = tf.browser.fromPixels(canvas);
    img = tf.image.resizeBilinear(img, [28, 28]);
    img = img.mean(2);
    img = img.expandDims(2);
    img = img.expandDims(0);
    img = img.toFloat().div(255.0);

    const preview = img.squeeze().mul(255);
    tf.browser.toPixels(preview.toInt(), miniCanvas);

    const prediction = model.predict(img);
    const probs = prediction.dataSync();
    const pIndex = prediction.argMax(1).dataSync()[0];
    const confidence = probs[pIndex];

    predictText.innerText = CLASS_NAMES[pIndex];
    confidenceText.innerText = `${Math.round(confidence * 100)}% confident`;

    if (roundActive && !roundEnding && pIndex === targetClass && confidence >= CONFIDENCE_THRESHOLD) {
      endRound(true);
    }
  });
}

// CLEAR
clearBtn.addEventListener("click", () => {
  resetCanvas();
  predictText.innerText = "?";
  confidenceText.innerText = "";
});

// GAME FLOW
startBtn.addEventListener("click", () => {
  score = 0;
  scoreText.innerText = score;
  startBtn.innerText = "Restart Game";
  startRound();
});

skipBtn.addEventListener("click", () => {
  if (!roundActive || roundEnding) return;
  endRound(false, true);
});

function pickTargetClass() {
  let next = targetClass;
  while (next === targetClass) {
    next = Math.floor(Math.random() * CLASS_NAMES.length);
  }
  return next;
}

function startRound() {
  roundEnding = false;
  targetClass = pickTargetClass();
  promptText.innerText = CLASS_NAMES[targetClass];
  resetCanvas();
  predictText.innerText = "?";
  confidenceText.innerText = "";
  hideOverlay();

  timeLeft = ROUND_SECONDS;
  timerText.innerText = timeLeft;
  timerBar.style.width = "100%";
  timerBar.style.background = "linear-gradient(90deg, #22d3ee, #a855f7)";

  roundActive = true;

  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timeLeft -= 1;
    timerText.innerText = Math.max(timeLeft, 0);
    timerBar.style.width = `${Math.max((timeLeft / ROUND_SECONDS) * 100, 0)}%`;

    if (timeLeft <= 5) {
      timerBar.style.background = "linear-gradient(90deg, #f87171, #f97316)";
    }

    if (timeLeft <= 0) {
      endRound(false);
    }
  }, 1000);
}

function endRound(success, skipped = false) {
  roundEnding = true;
  roundActive = false;
  clearInterval(timerInterval);

  if (success) {
    score += 1;
    scoreText.innerText = score;
    showOverlay(`Nailed it! It was ${CLASS_NAMES[targetClass]}`, "success");
  } else if (skipped) {
    showOverlay(`Skipped — it was ${CLASS_NAMES[targetClass]}`, "fail");
  } else {
    showOverlay(`Time's up! It was ${CLASS_NAMES[targetClass]}`, "fail");
  }

  setTimeout(() => {
    startRound();
  }, 1400);
}

function showOverlay(text, kind) {
  resultText.innerText = text;
  resultOverlay.className = `result-overlay ${kind}`;
}

function hideOverlay() {
  resultOverlay.className = "result-overlay hidden";
}
