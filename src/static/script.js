const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

let selectedFile = null;

function setStatus(msg, isError = false, spinner = false) {
  if (!msg) {
    statusEl.hidden = true;
    statusEl.innerHTML = "";
    statusEl.classList.remove("error");
    return;
  }
  statusEl.hidden = false;
  statusEl.classList.toggle("error", isError);
  statusEl.innerHTML = (spinner ? '<span class="spinner"></span>' : "") + msg;
}

function pickFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please pick an image file.", true);
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  preview.hidden = false;
  resultEl.hidden = true;
  setStatus("");
}

fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) pickFile(f);
});

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  preview.hidden = true;
  resultEl.hidden = true;
  setStatus("");
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("No image selected.", true);
    return;
  }
  analyzeBtn.disabled = true;
  setStatus("Analyzing image…", false, true);
  try {
    const fd = new FormData();
    fd.append("image", selectedFile);
    const r = await fetch("/predict", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    renderResult(data);
    setStatus("");
  } catch (err) {
    setStatus("Failed: " + err.message, true);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResult(data) {
  resultEl.hidden = false;
  document.getElementById("resultImg").src = data.image_url;

  const top = data.top || (data.predictions && data.predictions[0]);
  document.getElementById("dishName").textContent = top ? top.label : "Unknown";

  const conf = top ? top.confidence : 0;
  document.getElementById("confFill").style.width = (conf * 100).toFixed(1) + "%";
  document.getElementById("confValue").textContent = (conf * 100).toFixed(1) + "% confidence";

  const meta = data.meta || {};
  const metaRow = document.getElementById("metaRow");
  metaRow.innerHTML = "";
  ["diet", "course", "region"].forEach((k) => {
    if (meta[k] && meta[k] !== "unknown") {
      const chip = document.createElement("span");
      chip.className = "meta-chip";
      chip.textContent = meta[k];
      metaRow.appendChild(chip);
    }
  });

  if (data.is_ood) {
    metaRow.innerHTML = '<span class="meta-chip" style="background:#fbe9e7;color:#c0392b">' +
      (data.message || "Low confidence") + "</span>";
  }

  // nutrition
  const ng = document.getElementById("nutritionGrid");
  ng.innerHTML = "";
  const nutri = data.nutrition || {};
  const order = ["kcal", "carb", "protein", "fat"];
  let any = false;
  order.forEach((k) => {
    const v = nutri[k];
    if (!v) return;
    any = true;
    const card = document.createElement("div");
    card.className = "nutri-card";
    card.innerHTML =
      `<div class="label">${v.label}</div>` +
      `<div class="point">${v.point}<span class="unit">${v.unit}</span></div>` +
      `<div class="range">range ${v.low}–${v.high} ${v.unit}</div>`;
    ng.appendChild(card);
  });
  if (!any) {
    ng.innerHTML = '<div class="muted small">No nutrition data available for this dish.</div>';
  }
  const prov = (data.meta && data.meta.provenance) || {};
  const provText = prov.nutrition_source ? `Source: ${prov.nutrition_source} · ${prov.match_method || ""}` : "";
  document.getElementById("nutritionProvenance").textContent = provText;

  // allergens
  const al = document.getElementById("allergenList");
  al.innerHTML = "";
  const allergens = data.allergens || [];
  if (!allergens.length) {
    al.innerHTML = '<div class="muted small">No allergen information available.</div>';
  } else {
    allergens.forEach((a) => {
      const row = document.createElement("div");
      row.className = `allergen code-${a.code}`;
      row.innerHTML =
        `<span class="dot dot-${a.code}"></span>` +
        `<div class="text"><div class="name">${a.label}</div><div class="desc">${a.description}</div></div>`;
      al.appendChild(row);
    });
  }

  // top-k
  const tk = document.getElementById("topkList");
  tk.innerHTML = "";
  (data.predictions || []).slice(1).forEach((p) => {
    const li = document.createElement("li");
    li.innerHTML = `${p.label} <span class="conf">${(p.confidence * 100).toFixed(1)}%</span>`;
    tk.appendChild(li);
  });

  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}
