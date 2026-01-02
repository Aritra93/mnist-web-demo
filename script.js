let session;
let imageLoaded = false;

async function loadModel() {
  session = await ort.InferenceSession.create("mnist.onnx");
  console.log("ONNX model loaded");
}

loadModel();

const upload = document.getElementById("imageUpload");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

upload.addEventListener("change", event => {
  const file = event.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    // Clear canvas
    ctx.clearRect(0, 0, 28, 28);

    // Draw resized image
    ctx.drawImage(img, 0, 0, 28, 28);
    imageLoaded = true;
  };
  img.src = URL.createObjectURL(file);
});

async function predict() {
  if (!imageLoaded) {
    alert("Please upload an image first.");
    return;
  }

  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = new Float32Array(28 * 28);

  for (let i = 0; i < imageData.data.length; i += 4) {
    // Convert to grayscale
    const r = imageData.data[i];
    const g = imageData.data[i + 1];
    const b = imageData.data[i + 2];

    let gray = (r + g + b) / 3.0;

    // Normalize to [0,1]
    gray = gray / 255.0;

    // Invert if trained on MNIST (white digit on black bg)
    gray = 1.0 - gray;

    data[i / 4] = gray;
  }

  const tensor = new ort.Tensor("float32", data, [1, 1, 28, 28]);
  const output = await session.run({ input: tensor });

  const logits = output.logits.data;
  const prediction = logits.indexOf(Math.max(...logits));

  document.getElementById("result").innerText =
    `Predicted Digit: ${prediction}`;
}
