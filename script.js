let session = null;
let imageLoaded = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function loadModel() {
  session = await ort.InferenceSession.create("mnist.onnx");

  console.log("Model loaded");
  console.log("Inputs:", session.inputNames);
  console.log("Outputs:", session.outputNames);

  document.getElementById("predictBtn").disabled = false;
  document.getElementById("predictBtn").innerText = "Predict";
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", e => {
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, 28, 28);
    ctx.drawImage(img, 0, 0, 28, 28);
    imageLoaded = true;
  };
  img.src = URL.createObjectURL(e.target.files[0]);
});

async function predict() {
  if (!session || !imageLoaded) return;

  const imgData = ctx.getImageData(0, 0, 28, 28);
  const data = new Float32Array(28 * 28);

  for (let i = 0; i < imgData.data.length; i += 4) {
    let gray = (imgData.data[i] +
                imgData.data[i + 1] +
                imgData.data[i + 2]) / 3.0;
    data[i / 4] = 1.0 - gray / 255.0;
  }

  const tensor = new ort.Tensor("float32", data, [1, 1, 28, 28]);

  const feeds = {};
  feeds[session.inputNames[0]] = tensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  const pred = output.indexOf(Math.max(...output));
  document.getElementById("result").innerText =
    `Predicted Digit: ${pred}`;
}
