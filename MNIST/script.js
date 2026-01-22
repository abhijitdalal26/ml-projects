// getting elements canvas, predixtion text, clearBtn
const canvas = document.getElementById("main-canvas");
const ctx = canvas.getContext("2d");
const miniCanvas = document.getElementById("mini-canvas");
const miniCtx = miniCanvas.getContext("2d");

const predictText = document.getElementById("prediction-text");
const clearBtn = document.getElementById("clear-btn");

let model;

// LOAD THE MODEL (Async) >> means loading in background
async function loadModel() {
  try {
    console.log("Loading model...");
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded successfully!");
  } catch (error) {
    console.error("Model failed to load:", error);
    predictText.innerText = "Error";
  }
}
loadModel();

// DRAWING SETTINGS
let isDrawing = false;
ctx.lineWidth = 20; // Thickness
ctx.lineCap = "round";
ctx.strokeStyle = "white";
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height); // Initial black background

// 4. DRAWING EVENTS
// converts screen coordinates -> canvas coordinates.
const getPos = (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX || e.touches[0].clientX) - rect.left;
  const y = (e.clientY || e.touches[0].clientY) - rect.top;
  return { x, y };
};

const startDrawing = (e) => {
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

  // Trigger real-time prediction
  predictRealTime();
};

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);

// 5. THE PREDICTION PIPELINE
function predictRealTime() {
  if (!model) return;

  // tf.tidy cleans up memory (prevents browser lag)
  tf.tidy(() => {
    // Step 1: Capture pixels from canvas
    let img = tf.browser.fromPixels(canvas);

    // Step 2: Pre-process (Resize to 28x28, Grayscale, Normalize)
    img = tf.image.resizeBilinear(img, [28, 28]);
    img = img.mean(2); // RGB (3 channels) -> Grayscale (1 channel)
    img = img.expandDims(2); // Shape: (28, 28, 1)
    img = img.expandDims(0); // Shape: (1, 28, 28, 1) - Batch dimension
    img = img.toFloat().div(255.0); // Normalize to [0, 1]

    // Step 3: Show the 28x28 debug preview
    // We take the 28x28 grayscale image and put it on the mini-canvas
    const preview = img.squeeze().mul(255);
    tf.browser.toPixels(preview.toInt(), miniCanvas);

    // Step 4: RUN INFERENCE
    const prediction = model.predict(img);
    const pIndex = prediction.argMax(1).dataSync()[0];

    // Step 5: UPDATE UI
    predictText.innerText = pIndex;
  });
}

// 6. CLEAR LOGIC
clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  miniCtx.fillStyle = "black";
  miniCtx.fillRect(0, 0, 28, 28);
  predictText.innerText = "?";
});
