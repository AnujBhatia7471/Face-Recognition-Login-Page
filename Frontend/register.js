const API_BASE = "http://127.0.0.1:5000";


const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const sampleBtn = document.getElementById("sampleBtn");
const startBtn = document.getElementById("startBtn");

let sampleCount = 0;
const MAX_SAMPLES = 5;

async function startRegistration() {
  if (!email.value || !password.value) {
    result.innerText = "Email and password required";
    return;
  }

  await startCamera();
  sampleBtn.disabled = false;
  startBtn.disabled = true;
  result.innerText = "Camera started";
}

function takeSample() {
  if (sampleCount >= MAX_SAMPLES) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const f = new FormData();
    f.append("email", email.value.trim());
    f.append("password", password.value);
    f.append("image", blob);

    fetch(`${API_BASE}/register`, { method: "POST", body: f })
      .then(r => r.json())
      .then(d => {
        result.innerText = d.msg;
        if (!d.success) return;

        sampleCount++;
        if (d.completed) {
          stopCamera();
          sampleBtn.disabled = true;
        } else {
          sampleBtn.innerText = `Take Sample (${sampleCount + 1} / 5)`;
        }
      });
  }, "image/jpeg", 0.95);
}
