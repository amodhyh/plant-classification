/**
 * app.js — Plant Leaf Classifier
 * Handles drag-and-drop, file selection, image preview,
 * AJAX classify request, and dynamic results rendering.
 */

"use strict";

// ── DOM references ─────────────────────────────────────────────────────────────
const dropZone        = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const imagePreview    = document.getElementById("imagePreview");
const fileInput       = document.getElementById("fileInput");
const browseBtn       = document.getElementById("browseBtn");
const classifyBtn     = document.getElementById("classifyBtn");
const classifyBtnText = document.getElementById("classifyBtnText");
const classifySpinner = document.getElementById("classifySpinner");
const resetBtn        = document.getElementById("resetBtn");
const errorAlert      = document.getElementById("errorAlert");
const resultsSection  = document.getElementById("resultsSection");
const resultImage     = document.getElementById("resultImage");
const agreementBadge  = document.getElementById("agreementBadge");
const modelCardsRow   = document.getElementById("modelCardsRow");

// ── State ──────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Model meta (name → CSS class + accent color) ──────────────────────────────
const MODEL_META = {
  "InceptionV3":    { cls: "model-inception",   color: "#63ca84", icon: "🔬" },
  "ConvNeXt-Base":  { cls: "model-convnext",    color: "#36cfc9", icon: "🧩" },
  "EfficientNet":   { cls: "model-efficientnet",color: "#9d7aff", icon: "⚡" },
};

// ── Drag-and-drop ──────────────────────────────────────────────────────────────
["dragenter", "dragover"].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove("drag-over"); })
);
dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

// ── Click to open file dialog ──────────────────────────────────────────────────
dropZone.addEventListener("click",  () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
browseBtn.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});

// ── Reset ──────────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", resetApp);

// ── Classify ──────────────────────────────────────────────────────────────────
classifyBtn.addEventListener("click", classify);

// ── File handling ──────────────────────────────────────────────────────────────
function handleFileSelect(file) {
  const allowed = ["image/jpeg", "image/png"];
  if (!allowed.includes(file.type)) {
    showError("Invalid file type. Please upload a JPG or PNG image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("File is too large. Maximum allowed size is 10 MB.");
    return;
  }

  selectedFile = file;
  hideError();

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    imagePreview.src = e.target.result;
    imagePreview.classList.remove("d-none");
    dropZoneContent.classList.add("d-none");
    dropZone.classList.add("has-image");
  };
  reader.readAsDataURL(file);

  classifyBtn.disabled = false;
  resetBtn.classList.remove("d-none");
}

// ── Classify request ───────────────────────────────────────────────────────────
async function classify() {
  if (!selectedFile) return;

  // UI: loading state
  classifyBtn.disabled = true;
  classifyBtnText.textContent = "Classifying…";
  classifySpinner.classList.remove("d-none");
  hideError();
  resultsSection.classList.add("d-none");
  modelCardsRow.innerHTML = "";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/classify", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok || !data.success) {
      showError(data.error || "An unexpected error occurred. Please try again.");
      return;
    }

    renderResults(data);

  } catch (err) {
    console.error("Classify fetch error:", err);
    showError("Network error — could not reach the server. Is Flask running?");
  } finally {
    classifyBtn.disabled = false;
    classifyBtnText.textContent = "🔍 Classify";
    classifySpinner.classList.add("d-none");
  }
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  // Uploaded image
  resultImage.src = imagePreview.src;

  // Agreement badge
  agreementBadge.classList.remove("d-none", "agree", "disagree");
  if (data.models_agree) {
    agreementBadge.classList.add("agree");
    agreementBadge.innerHTML = "✅ All models agree";
  } else {
    agreementBadge.classList.add("disagree");
    agreementBadge.innerHTML = "⚠️ Models disagree";
  }

  // Model cards
  const modelOrder = ["InceptionV3", "ConvNeXt-Base", "EfficientNet"];
  modelOrder.forEach((name, idx) => {
    const result = data.results[name];
    const meta   = MODEL_META[name] || { cls: "", color: "#8b949e", icon: "🤖" };
    const col    = document.createElement("div");
    col.className = "col-12 col-md-6 col-lg-4";
    col.innerHTML = buildModelCard(name, result, meta, idx);
    modelCardsRow.appendChild(col);
  });

  // Animate bars after render
  requestAnimationFrame(() => animateBars());

  // Show section
  resultsSection.classList.remove("d-none");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Build model card HTML ──────────────────────────────────────────────────────
function buildModelCard(name, result, meta, idx) {
  const delay = (idx * 0.1).toFixed(2);

  if (result.error) {
    return `
      <div class="model-card has-error ${meta.cls}" style="animation-delay:${delay}s">
        <div class="card-model-name">
          <span class="card-model-dot" style="background:${meta.color}"></span>
          ${meta.icon} ${name}
        </div>
        <p class="card-error-msg">⚠️ ${escapeHtml(result.error)}</p>
      </div>`;
  }

  const probs = result.probabilities; // already sorted desc by backend
  const barsHtml = Object.entries(probs).map(([cls, prob], i) => {
    const pct     = (prob * 100).toFixed(1);
    const isTop   = i === 0;
    const barClr  = isTop ? meta.color : "rgba(255,255,255,0.12)";
    return `
      <div>
        <div class="prob-label">
          <span class="cls-name">${escapeHtml(cls)}</span>
          <span class="cls-pct">${pct}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill"
               data-width="${pct}"
               style="width:0%;background:${barClr}">
          </div>
        </div>
      </div>`;
  }).join("");

  return `
    <div class="model-card ${meta.cls}" style="animation-delay:${delay}s">
      <div class="card-model-name">
        <span class="card-model-dot" style="background:${meta.color}"></span>
        ${meta.icon} ${name}
      </div>
      <div class="card-prediction">${escapeHtml(result.predicted_class)}</div>
      <div class="card-confidence">${result.confidence}% confidence</div>
      <div class="card-divider"></div>
      <div class="prob-bars">${barsHtml}</div>
    </div>`;
}

// ── Animate probability bars ───────────────────────────────────────────────────
function animateBars() {
  document.querySelectorAll(".prob-bar-fill[data-width]").forEach(bar => {
    const target = bar.getAttribute("data-width");
    // Small delay to let CSS transition work visually
    setTimeout(() => { bar.style.width = target + "%"; }, 50);
  });
}

// ── Reset app ──────────────────────────────────────────────────────────────────
function resetApp() {
  selectedFile = null;
  fileInput.value = "";
  imagePreview.src = "#";
  imagePreview.classList.add("d-none");
  dropZoneContent.classList.remove("d-none");
  dropZone.classList.remove("has-image", "drag-over");
  classifyBtn.disabled = true;
  classifyBtnText.textContent = "🔍 Classify";
  classifySpinner.classList.add("d-none");
  resetBtn.classList.add("d-none");
  resultsSection.classList.add("d-none");
  modelCardsRow.innerHTML = "";
  agreementBadge.classList.add("d-none");
  hideError();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Error helpers ──────────────────────────────────────────────────────────────
function showError(msg) {
  errorAlert.textContent = "❌  " + msg;
  errorAlert.classList.remove("d-none");
}
function hideError() {
  errorAlert.classList.add("d-none");
  errorAlert.textContent = "";
}

// ── XSS-safe escaping ─────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
