const state = {
  selectedPPE: new Set(),
  defaultPPE: [],
  config: null,
  stream: null,
  previewTimer: null,
  previewPlaying: false,
};

const modelSelect = document.getElementById("model-select");
const confidenceThreshold = document.getElementById("confidence-threshold");
const complianceThreshold = document.getElementById("compliance-threshold");
const temporalWindow = document.getElementById("temporal-window");
const confidenceValue = document.getElementById("confidence-value");
const complianceValue = document.getElementById("compliance-value");
const temporalValue = document.getElementById("temporal-value");
const ppeChips = document.getElementById("ppe-chips");
const modelStatus = document.getElementById("model-status");
const runStatus = document.getElementById("run-status");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const metricsGrid = document.getElementById("metrics-grid");
const mediaHost = document.getElementById("media-host");
const peopleTable = document.getElementById("people-table");
const eventsTable = document.getElementById("events-table");
const timelineCard = document.getElementById("timeline-card");
const downloadLinks = document.getElementById("download-links");

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function fixed(value, digits = 2, fallback = "0.00") {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : fallback;
}

function setSectionMessage(element, message) {
  element.innerHTML = `<p class="inline-note">${message}</p>`;
}

function setSliderLabels() {
  confidenceValue.textContent = Number(confidenceThreshold.value).toFixed(2);
  complianceValue.textContent = Number(complianceThreshold.value).toFixed(2);
  temporalValue.textContent = `${temporalWindow.value} frames`;
}

function collectSettings() {
  return {
    model_name: modelSelect.value,
    required_ppe: JSON.stringify([...state.selectedPPE]),
    confidence_threshold: confidenceThreshold.value,
    compliance_threshold: complianceThreshold.value,
    temporal_window: temporalWindow.value,
  };
}

function setRunStatus(message, active = false) {
  runStatus.textContent = message;
  runStatus.className = `status-pill${active ? "" : " muted"}`;
}

function renderPPEChips(names) {
  ppeChips.innerHTML = "";
  names.forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip ${state.selectedPPE.has(name) ? "active" : ""}`;
    button.textContent = name;
    button.addEventListener("click", () => {
      if (state.selectedPPE.has(name)) {
        state.selectedPPE.delete(name);
      } else {
        state.selectedPPE.add(name);
      }
      renderPPEChips(names);
    });
    ppeChips.appendChild(button);
  });
}

async function loadConfig(modelName = null) {
  const url = modelName ? `/api/config?model_name=${encodeURIComponent(modelName)}` : "/api/config";
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to load configuration.");
  }

  state.config = payload;
  state.defaultPPE = payload.default_required_ppe || [];

  modelSelect.innerHTML = payload.available_models
    .map((name) => `<option value="${name}" ${name === payload.selected_model ? "selected" : ""}>${name}</option>`)
    .join("");

  if (state.selectedPPE.size === 0 || modelName) {
    state.selectedPPE = new Set(payload.default_required_ppe || []);
  } else {
    const allowed = new Set(payload.ppe_classes);
    state.selectedPPE = new Set([...state.selectedPPE].filter((name) => allowed.has(name)));
    if (state.selectedPPE.size === 0) {
      state.selectedPPE = new Set(payload.default_required_ppe || []);
    }
  }

  confidenceThreshold.value = payload.defaults.confidence_threshold;
  complianceThreshold.value = payload.defaults.compliance_threshold;
  temporalWindow.value = payload.defaults.temporal_window;
  setSliderLabels();
  renderPPEChips(payload.ppe_classes);
  modelStatus.textContent = `Loaded ${payload.selected_model} with ${payload.class_names.length} classes`;
}

function createMetricCard(label, value) {
  return `
    <article class="metric-card">
      <span>${label}</span>
      <strong>${value}</strong>
    </article>
  `;
}

function renderMetrics(summary, mode) {
  const safeSummary = summary || {};
  const cards = [];
  if (mode === "image") {
    cards.push(createMetricCard("Workers", asNumber(safeSummary.workers, 0)));
    cards.push(createMetricCard("Compliant", asNumber(safeSummary.compliant, 0)));
    cards.push(createMetricCard("At Risk", asNumber(safeSummary.at_risk, 0)));
    cards.push(createMetricCard("Violations", asNumber(safeSummary.violations, 0)));
    cards.push(createMetricCard("Adaptive Score", fixed(safeSummary.mean_adaptive_score, 2)));
    cards.push(createMetricCard("Entropy", fixed(safeSummary.mean_entropy, 3, "0.000")));
  } else {
    cards.push(createMetricCard("Frames", asNumber(safeSummary.frames_processed, 0)));
    cards.push(createMetricCard("Duration", `${fixed(safeSummary.duration_seconds, 1, "0.0")}s`));
    cards.push(createMetricCard("Unique Workers", asNumber(safeSummary.unique_workers_detected, 0)));
    cards.push(createMetricCard("Avg Workers / Frame", fixed(safeSummary.average_workers_per_frame, 2)));
    cards.push(createMetricCard("Compliance Rate", `${fixed(asNumber(safeSummary.observation_compliance_rate, 0) * 100, 1, "0.0")}%`));
    cards.push(createMetricCard("Peak Violations", asNumber(safeSummary.peak_concurrent_violations, 0)));
    cards.push(createMetricCard("Event Count", asNumber(safeSummary.event_count, 0)));
  }
  metricsGrid.innerHTML = cards.join("");
}

function renderMedia(result) {
  clearPreviewTimer();
  downloadLinks.innerHTML = `
    <a href="${result.annotated_media_url}" target="_blank" rel="noreferrer">${result.mode === "video" ? "Annotated video" : "Annotated image"}</a>
    <a href="${result.report_url}" target="_blank" rel="noreferrer">JSON report</a>
    ${result.events_csv_url ? `<a href="${result.events_csv_url}" target="_blank" rel="noreferrer">CSV events</a>` : ""}
  `;

  if (result.mode === "video") {
    mediaHost.innerHTML = `
      <div class="preview-stack">
        <div class="browser-preview-card">
          <div class="preview-header">
            <strong>Browser Preview</strong>
          </div>
          <div class="preview-stage">
            <img id="browser-preview-frame" class="annotated-image" alt="Processed video preview">
          </div>
          <div class="preview-controls">
            <button type="button" class="secondary-button" id="preview-toggle">Pause</button>
            <button type="button" class="ghost-button" id="preview-restart">Play from Start</button>
            <input type="range" id="preview-slider" min="0" max="${Math.max((result.browser_preview?.frame_urls?.length || 1) - 1, 0)}" step="1" value="0">
            <span id="preview-frame-label">Frame 1 / ${result.browser_preview?.frame_urls?.length || 1}</span>
          </div>
        </div>
      </div>
    `;
    setupBrowserPreview(result.browser_preview);
  } else {
    mediaHost.innerHTML = `<img class="annotated-image" src="${result.annotated_media_url}" alt="Annotated PPE result">`;
  }
}

function clearPreviewTimer() {
  if (state.previewTimer) {
    window.clearInterval(state.previewTimer);
    state.previewTimer = null;
  }
  state.previewPlaying = false;
}

function setupBrowserPreview(preview) {
  clearPreviewTimer();
  if (!preview || !preview.frame_urls || preview.frame_urls.length === 0) {
    return;
  }

  const image = document.getElementById("browser-preview-frame");
  const toggleButton = document.getElementById("preview-toggle");
  const restartButton = document.getElementById("preview-restart");
  const slider = document.getElementById("preview-slider");
  const label = document.getElementById("preview-frame-label");
  const frameUrls = preview.frame_urls;
  const frameDelay = Math.max(40, Math.round(1000 / Math.max(preview.fps || 8, 1)));

  const showFrame = (index) => {
    const safeIndex = Math.max(0, Math.min(index, frameUrls.length - 1));
    slider.value = String(safeIndex);
    image.src = frameUrls[safeIndex];
    label.textContent = `Frame ${safeIndex + 1} / ${frameUrls.length}`;
  };

  const syncToggleLabel = () => {
    toggleButton.textContent = state.previewPlaying ? "Pause" : "Play";
  };

  const stopPlayback = () => {
    clearPreviewTimer();
    syncToggleLabel();
  };

  const startPlayback = (fromStart = false) => {
    if (fromStart) {
      showFrame(0);
    }
    if (frameUrls.length <= 1) {
      state.previewPlaying = false;
      syncToggleLabel();
      return;
    }
    clearPreviewTimer();
    state.previewPlaying = true;
    syncToggleLabel();
    state.previewTimer = window.setInterval(() => {
      const nextIndex = Number(slider.value) + 1;
      if (nextIndex >= frameUrls.length) {
        stopPlayback();
        return;
      }
      showFrame(nextIndex);
    }, frameDelay);
  };

  toggleButton.onclick = () => {
    if (state.previewPlaying) {
      stopPlayback();
      return;
    }

    if (Number(slider.value) >= frameUrls.length - 1) {
      startPlayback(true);
      return;
    }

    startPlayback();
  };

  restartButton.onclick = () => {
    startPlayback(true);
  };

  slider.oninput = () => {
    stopPlayback();
    showFrame(Number(slider.value));
  };

  showFrame(0);
  syncToggleLabel();
  startPlayback();
}

function renderStatusBadge(kind, label) {
  return `<span class="status-badge ${kind}">${label}</span>`;
}

function renderPeople(result) {
  const people = asArray(result.people);
  if (people.length === 0) {
    setSectionMessage(peopleTable, "No worker-level results were available for this run.");
    return;
  }

  if (result.mode === "video") {
    peopleTable.innerHTML = `
      <table class="table">
        <thead>
          <tr>
            <th>Track</th>
            <th>Frames Seen</th>
            <th>Avg Adaptive Score</th>
            <th>Dominant Status</th>
            <th>Persistent Missing PPE</th>
          </tr>
        </thead>
        <tbody>
          ${people
            .map(
              (person) => `
              <tr>
                <td>${person.track_label || "Unknown"}</td>
                <td>${asNumber(person.frames_seen, 0)}</td>
                <td>${fixed(person.average_adaptive_score, 2)}</td>
                <td>${renderStatusBadge(person.dominant_status || "warning", String(person.dominant_status || "unknown").toUpperCase())}</td>
                <td>${asArray(person.persistent_missing).length ? asArray(person.persistent_missing).join(", ") : "None"}</td>
              </tr>
            `
            )
            .join("")}
        </tbody>
      </table>
    `;
    return;
  }

  peopleTable.innerHTML = `
    <table class="table">
      <thead>
        <tr>
          <th>Worker</th>
          <th>Status</th>
          <th>Detected PPE</th>
          <th>Missing PPE</th>
          <th>Adaptive Score</th>
        </tr>
      </thead>
      <tbody>
        ${people
          .map(
            (person) => `
            <tr>
              <td>${person.track_label || "Unknown"}</td>
              <td>${renderStatusBadge(person.status_kind || "warning", String(person.status_label || "UNKNOWN").toUpperCase())}</td>
              <td>${asArray(person.found_now).length ? asArray(person.found_now).join(", ") : "None"}</td>
              <td>${asArray(person.missing_now).length ? asArray(person.missing_now).join(", ") : "None"}</td>
              <td>${fixed(person.adaptive_score, 2)}</td>
            </tr>
          `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderTimeline(result) {
  const timeline = asArray(result.timeline);
  if (result.mode !== "video" || timeline.length === 0) {
    setSectionMessage(timelineCard, "Temporal samples appear here after a video run.");
    return;
  }

  const maxViolations = Math.max(...timeline.map((point) => asNumber(point.violations, 0)), 1);
  timelineCard.innerHTML = `
    <div class="timeline-strip">
      ${timeline
        .map((point) => {
          const height = Math.max(10, asNumber(point.mean_adaptive_score, 0) * 100);
          const violationWidth = `${(asNumber(point.violations, 0) / maxViolations) * 100}%`;
          return `
            <div class="timeline-step">
              <div class="timeline-bar" style="height:${height}px"></div>
              <div class="timeline-violations"><span style="width:${violationWidth}"></span></div>
              <div class="timeline-label">${fixed(point.time_seconds, 1, "0.0")}s</div>
            </div>
          `;
        })
        .join("")}
    </div>
    <p class="inline-note">Bar height reflects mean adaptive compliance score. Red markers reflect violation density at each sampled second.</p>
  `;
}

function renderEvents(result) {
  const events = asArray(result.events);
  if (result.mode !== "video") {
    setSectionMessage(eventsTable, "Violation segments are generated for video analysis runs.");
    return;
  }

  if (events.length === 0) {
    setSectionMessage(eventsTable, "No sustained violation segments were detected in the uploaded clip.");
    return;
  }

  eventsTable.innerHTML = `
    <table class="table">
      <thead>
        <tr>
          <th>Track</th>
          <th>Start</th>
          <th>End</th>
          <th>Duration</th>
          <th>Missing PPE</th>
        </tr>
      </thead>
      <tbody>
        ${events
          .map(
            (event) => `
            <tr>
              <td>${event.track_label || "Unknown"}</td>
              <td>${fixed(event.start_time_seconds, 1, "0.0")}s</td>
              <td>${fixed(event.end_time_seconds, 1, "0.0")}s</td>
              <td>${fixed(event.duration_seconds, 1, "0.0")}s</td>
              <td>${asArray(event.missing_ppe).length ? asArray(event.missing_ppe).join(", ") : "None"}</td>
            </tr>
          `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderResults(result) {
  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  setRunStatus(`Completed ${result.mode} analysis`, true);
  try {
    renderMetrics(result.summary, result.mode);
  } catch (error) {
    metricsGrid.innerHTML = createMetricCard("Render Error", "Check console");
  }
  try {
    renderMedia(result);
  } catch (error) {
    mediaHost.innerHTML = `<p class="inline-note">Could not render media preview.</p>`;
  }
  try {
    renderPeople(result);
  } catch (error) {
    setSectionMessage(peopleTable, "Could not render worker breakdown for this run.");
  }
  try {
    renderTimeline(result);
  } catch (error) {
    setSectionMessage(timelineCard, "Could not render timeline for this run.");
  }
  try {
    renderEvents(result);
  } catch (error) {
    setSectionMessage(eventsTable, "Could not render event summary for this run.");
  }
}

async function submitFile(endpoint, file, statusMessage) {
  if (!file) {
    throw new Error("Please choose a file first.");
  }

  const formData = new FormData();
  formData.append("file", file);
  const settings = collectSettings();
  Object.entries(settings).forEach(([key, value]) => formData.append(key, value));

  setRunStatus(statusMessage, true);
  const response = await fetch(endpoint, { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Analysis failed.");
  }
  renderResults(payload);
}

async function handleImageSubmit(event) {
  event.preventDefault();
  const file = document.getElementById("image-file").files[0];
  await submitFile("/api/analyze/image", file, "Running image analysis...");
}

async function handleVideoSubmit(event) {
  event.preventDefault();
  const file = document.getElementById("video-file").files[0];
  await submitFile("/api/analyze/video", file, "Processing video. This can take a little while...");
}

async function startCamera() {
  if (state.stream) {
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  state.stream = stream;
  const video = document.getElementById("camera-stream");
  video.srcObject = stream;
  setRunStatus("Camera ready", true);
}

function stopCamera() {
  if (!state.stream) {
    return;
  }
  state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  document.getElementById("camera-stream").srcObject = null;
  setRunStatus("Camera stopped", false);
}

async function captureCamera() {
  const video = document.getElementById("camera-stream");
  if (!state.stream || video.videoWidth === 0 || video.videoHeight === 0) {
    throw new Error("Start the camera before capturing.");
  }

  const canvas = document.getElementById("camera-canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
  const snapshot = new File([blob], "camera_snapshot.jpg", { type: "image/jpeg" });
  await submitFile("/api/analyze/image", snapshot, "Analyzing camera snapshot...");
}

function initTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tab;
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.toggle("active", item === button));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === target));
    });
  });
}

function initEvents() {
  confidenceThreshold.addEventListener("input", setSliderLabels);
  complianceThreshold.addEventListener("input", setSliderLabels);
  temporalWindow.addEventListener("input", setSliderLabels);

  modelSelect.addEventListener("change", async () => {
    try {
      await loadConfig(modelSelect.value);
    } catch (error) {
      setRunStatus(error.message, false);
    }
  });

  document.getElementById("image-form").addEventListener("submit", async (event) => {
    try {
      await handleImageSubmit(event);
    } catch (error) {
      setRunStatus(error.message, false);
    }
  });

  document.getElementById("video-form").addEventListener("submit", async (event) => {
    try {
      await handleVideoSubmit(event);
    } catch (error) {
      setRunStatus(error.message, false);
    }
  });

  document.getElementById("reset-ppe").addEventListener("click", () => {
    state.selectedPPE = new Set(state.defaultPPE);
    renderPPEChips(state.config.ppe_classes);
  });

  document.getElementById("start-camera").addEventListener("click", async () => {
    try {
      await startCamera();
    } catch (error) {
      setRunStatus(error.message, false);
    }
  });

  document.getElementById("stop-camera").addEventListener("click", stopCamera);
  document.getElementById("capture-camera").addEventListener("click", async () => {
    try {
      await captureCamera();
    } catch (error) {
      setRunStatus(error.message, false);
    }
  });
}

async function init() {
  initTabs();
  initEvents();
  setSliderLabels();
  try {
    await loadConfig();
    setRunStatus("Ready for analysis", false);
  } catch (error) {
    setRunStatus(error.message, false);
  }
}

init();
