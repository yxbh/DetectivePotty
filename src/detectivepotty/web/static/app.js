const state = {
  events: [],
  dogs: [],
  selectedEventId: null,
  selectedLabel: "unknown",
};

const eventList = document.querySelector("#event-list");
const eventCount = document.querySelector("#event-count");
const detailPane = document.querySelector("#detail-pane");
const statusFilter = document.querySelector("#status-filter");
const cameraFilter = document.querySelector("#camera-filter");
const refreshButton = document.querySelector("#refresh-button");
const cardTemplate = document.querySelector("#event-card-template");

refreshButton.addEventListener("click", loadEvents);
statusFilter.addEventListener("change", loadEvents);
cameraFilter.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadEvents();
  }
});

document.addEventListener("keydown", (event) => {
  if (!state.selectedEventId) {
    return;
  }
  const labels = { "1": "pee", "2": "poop", "3": "not_potty", "0": "unknown" };
  if (labels[event.key]) {
    state.selectedLabel = labels[event.key];
    updateLabelButtons();
  }
});

init();

async function init() {
  await loadDogs();
  await loadEvents();
}

async function loadDogs() {
  try {
    const response = await fetch("/api/dogs");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    state.dogs = Array.isArray(data.dogs) ? data.dogs : [];
  } catch (error) {
    state.dogs = [];
  }
}

async function loadEvents() {
  eventList.innerHTML = '<div class="empty-state">Loading events…</div>';
  const params = new URLSearchParams({ limit: "200" });
  if (statusFilter.value) {
    params.set("label_status", statusFilter.value);
  }
  if (cameraFilter.value.trim()) {
    params.set("camera", cameraFilter.value.trim());
  }

  try {
    const response = await fetch(`/api/events?${params}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const unfiltered = response.headers.get("X-Unfiltered-Count");
    state.events = await response.json();
    updateEventCount(state.events.length, unfiltered);
    if (
      state.selectedEventId &&
      !state.events.some((item) => item.event_id === state.selectedEventId)
    ) {
      state.selectedEventId = null;
      detailPane.innerHTML = '<div class="empty-state">Select an event to review.</div>';
    }
    renderEventList();
    if (state.events.length && !state.selectedEventId) {
      selectEvent(state.events[0].event_id);
    }
  } catch (error) {
    eventList.innerHTML = `<div class="error-state">Failed to load events: ${error.message}</div>`;
    updateEventCount(0, null);
  }
}

function updateEventCount(shown, unfiltered) {
  if (!eventCount) {
    return;
  }
  let text = `${shown} event${shown === 1 ? "" : "s"}`;
  const total = unfiltered == null ? null : Number(unfiltered);
  if (total != null && !Number.isNaN(total) && total !== shown) {
    text += ` · ${total} total on disk`;
  }
  eventCount.textContent = text;
}

function renderEventList() {
  eventList.innerHTML = "";
  if (!state.events.length) {
    eventList.innerHTML = '<div class="empty-state">No events match the filters.</div>';
    return;
  }

  for (const item of state.events) {
    const card = cardTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.eventId = item.event_id;
    card.classList.toggle("active", item.event_id === state.selectedEventId);
    card.querySelector(".event-camera").textContent = item.camera || "Unknown camera";
    card.querySelector(".event-time").textContent = formatTime(item.utc_ts);
    const dogEl = card.querySelector(".event-dog");
    if (dogEl) {
      dogEl.textContent = item.dog ? `🐕 ${item.dog}` : "";
    }
    card.querySelector(".event-guess").textContent = formatGuess(item);
    card.querySelector(".event-status").textContent = item.label_status || "unlabeled";
    const thumb = card.querySelector(".event-thumb");
    if (item.thumbnail_url) {
      thumb.src = item.thumbnail_url;
    } else {
      thumb.removeAttribute("src");
      thumb.alt = "No thumbnail";
    }
    card.addEventListener("click", () => selectEvent(item.event_id));
    eventList.append(card);
  }
}

async function selectEvent(eventId) {
  state.selectedEventId = eventId;
  renderEventList();
  detailPane.innerHTML = '<div class="empty-state">Loading event…</div>';
  try {
    const response = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const detail = await response.json();
    renderDetail(detail);
  } catch (error) {
    detailPane.innerHTML = `<div class="error-state">Failed to load event: ${error.message}</div>`;
  }
}

function renderDetail(detail) {
  const meta = detail.metadata;
  const media = detail.media;
  state.selectedLabel = meta.label || "unknown";

  detailPane.innerHTML = `
    <h2>${escapeHtml(detail.summary.camera)} · ${formatTime(meta.utc_ts)}</h2>
    <div class="video-wrap">
      ${media.clip ? `<video controls src="${media.clip}"></video>` : '<div class="empty-state">No clip.mp4 found.</div>'}
    </div>
    <div id="hero-slot"></div>
    <div class="summary-grid">
      ${summaryCard("Guess", `${meta.classifier_guess || "unknown"} ${formatConfidence(meta.classifier_confidence)}`)}
      ${summaryCard("Current label", `${meta.label || "unknown"} / ${meta.label_status || "unlabeled"}`)}
      ${summaryCard("Dog", meta.dog || "unassigned")}
      ${summaryCard("Trigger", meta.trigger_reason || "unknown")}
      ${summaryCard("Latency", meta.trigger_latency_s == null ? "n/a" : `${meta.trigger_latency_s.toFixed(2)}s`)}
      ${summaryCard("Flags", flagsText(meta))}
      ${summaryCard("Posture", meta.extra?.posture_summary || "n/a")}
    </div>
    <section class="label-panel">
      <h3>Label this event</h3>
      <div class="label-buttons">
        ${labelButton("pee", "Pee (1)")}
        ${labelButton("poop", "Poop (2)")}
        ${labelButton("not_potty", "Not potty (3)")}
        ${labelButton("unknown", "Unknown (0)")}
      </div>
      <div class="form-row">
        <label>Dog
          <select id="dog-select">${dogOptions(meta.dog)}</select>
        </label>
        <label>Status
          <select id="label-status">
            ${option("labeled", meta.label_status)}
            ${option("rejected", meta.label_status)}
            ${option("uncertain", meta.label_status)}
          </select>
        </label>
        <label>Note
          <textarea id="label-note" placeholder="Optional training note">${escapeHtml(meta.extra?.label_note || "")}</textarea>
        </label>
        <button id="save-label" type="button">Save label</button>
      </div>
      <p id="save-status" class="muted"></p>
    </section>
    ${mediaStrip("Crops", media.crops)}
    ${mediaStrip("Pose overlay", media.crops_overlay || [])}
    ${mediaStrip("Frames", media.frames)}
    ${poseSection(meta)}
    <section class="metadata-panel">
      <h3>Metadata</h3>
      <pre>${escapeHtml(JSON.stringify(meta, null, 2))}</pre>
    </section>
  `;

  detailPane.querySelectorAll("[data-label]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedLabel = button.dataset.label;
      updateLabelButtons();
    });
  });
  detailPane.querySelector("#save-label").addEventListener("click", saveLabel);
  detailPane.querySelectorAll(".strip-grid img").forEach((img) => {
    img.addEventListener("click", () => showHeroImage(img.src, img.alt));
  });
  updateLabelButtons();
}

async function saveLabel() {
  const status = detailPane.querySelector("#label-status").value;
  const note = detailPane.querySelector("#label-note").value;
  const dogEl = detailPane.querySelector("#dog-select");
  const dog = dogEl ? dogEl.value || null : null;
  const saveStatus = detailPane.querySelector("#save-status");
  saveStatus.textContent = "Saving…";

  try {
    const response = await fetch(`/api/events/${encodeURIComponent(state.selectedEventId)}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: state.selectedLabel, label_status: status, note, dog }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const updated = await response.json();
    const index = state.events.findIndex((item) => item.event_id === updated.event_id);
    if (index !== -1) {
      state.events[index] = updated;
      renderEventList();
    }
    saveStatus.textContent = "Saved.";
  } catch (error) {
    saveStatus.textContent = `Save failed: ${error.message}`;
  }
}

function dogOptions(current) {
  const options = ['<option value="">Unassigned</option>'];
  const roster = state.dogs.slice();
  if (current && !roster.includes(current)) {
    roster.unshift(current);
  }
  for (const name of roster) {
    const selected = name === current ? " selected" : "";
    options.push(`<option value="${escapeHtml(name)}"${selected}>${escapeHtml(name)}</option>`);
  }
  return options.join("");
}

function updateLabelButtons() {
  detailPane.querySelectorAll("[data-label]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.label === state.selectedLabel);
  });
}

function showHeroImage(src, alt) {
  const hero = detailPane.querySelector("#hero-slot");
  hero.innerHTML = `<img class="hero-image" src="${src}" alt="${escapeHtml(alt)}" />`;
}

function mediaStrip(title, items) {
  if (!items.length) {
    return "";
  }
  const images = items
    .map((item) => `<img src="${item.url}" alt="${escapeHtml(item.name)}" loading="lazy" />`)
    .join("");
  return `<section class="media-strip"><h3>${title}</h3><div class="strip-grid">${images}</div></section>`;
}

function poseSection(meta) {
  const pose = meta.extra && meta.extra.pose;
  const features = pose && pose.features;
  if (!features) {
    return "";
  }
  const rows = [
    ["Spine angle", features.spine_angle_deg, "°"],
    ["Hip offset", features.hip_offset_ratio, ""],
    ["Tail angle", features.tail_angle_deg, "°"],
    ["Centroid motion", features.centroid_motion_ratio, ""],
    ["Dwell", features.dwell_duration_s, "s"],
    ["Coverage", features.coverage, ""],
  ];
  const cells = rows
    .filter(([, value]) => value != null)
    .map(([name, value, unit]) => summaryCard(name, `${Number(value).toFixed(2)}${unit}`))
    .join("");
  if (!cells) {
    return "";
  }
  const kpCount = pose.keypoints ? pose.keypoints.length : 0;
  return `<section class="pose-panel"><h3>Pose features <span class="muted">(${kpCount} posed frames)</span></h3><div class="summary-grid">${cells}</div></section>`;
}

function labelButton(value, text) {
  return `<button type="button" data-label="${value}">${text}</button>`;
}

function option(value, current) {
  const selected = value === current ? " selected" : "";
  return `<option value="${value}"${selected}>${value}</option>`;
}

function summaryCard(title, value) {
  return `<div class="summary-card"><span>${title}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function formatTime(value) {
  if (!value) {
    return "Unknown time";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function formatGuess(item) {
  const guess = item.classifier_guess || "unknown";
  return `${guess} ${formatConfidence(item.classifier_confidence)}`;
}

function formatConfidence(value) {
  return value == null ? "" : `(${Math.round(value * 100)}%)`;
}

function flagsText(meta) {
  const flags = [];
  if (meta.multi_dog) {
    flags.push("multi-dog");
  }
  if (meta.ambiguous) {
    flags.push("ambiguous");
  }
  return flags.length ? flags.join(", ") : "none";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" };
    return entities[char];
  });
}
