// Process Audio -- the browser front end. Plain ES module, no framework:
// everything on screen is built from GET /api/capabilities, so a new genre
// or option in the recipes shows up here without touching this file.
//
// The option rules (which options apply, what their default is, when a
// value counts as "changed") mirror build_argv() in pipeline_capabilities.py
// and BuildArgv() in the Go server; the server re-validates everything.

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// replaceChildren() would print a literal "null" for optional parts
function fill(el, ...kids) {
  el.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
}

const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// ------------------------------------------------------------------ api

const CSRF = { "X-Requested-With": "process-audio" };

async function api(method, path, body) {
  const opts = { method, headers: { ...(method === "GET" ? {} : CSRF) } };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.setRequestHeader("X-Requested-With", "process-audio");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error((data && data.error) || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("upload failed -- is the server still running?"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

// ------------------------------------------------------------------ formatting

function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${mm}:${String(ss).padStart(2, "0")}`;
}
function fmtBytes(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(1) + " GB";
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + " MB";
  if (n >= 1 << 10) return Math.round(n / (1 << 10)) + " KB";
  return n + " B";
}
function fmtNum(v, step) {
  if (typeof v !== "number") return String(v ?? "");
  const decimals = step && step < 1 ? Math.min(3, Math.ceil(-Math.log10(step) - 1e-9)) : 0;
  return String(Number(v.toFixed(decimals)));
}
function ago(iso) {
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

// ------------------------------------------------------------------ state

const state = {
  caps: null,
  pipeline: null,        // pipeline object
  upload: null,          // upload meta
  peaks: null,
  type: null,            // type id
  analysis: null,        // analysis payload for (pipeline, type)
  analysisFor: null,
  values: {},            // only options the user changed
  outputs: new Set(),
  formats: new Set(),
};

const banner = {
  show(msg) { $("#banner-text").textContent = msg; $("#banner").hidden = false; },
  hide() { $("#banner").hidden = true; },
};

// ------------------------------------------------------------------ option rules

function optDefault(o, type) {
  if (type && o.defaults_by_type && type in o.defaults_by_type) return o.defaults_by_type[type];
  return o.default;
}
function optValue(o) {
  return o.id in state.values ? state.values[o.id] : optDefault(o, state.type);
}
function same(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) return JSON.stringify(a ?? []) === JSON.stringify(b ?? []);
  return a === b || ((a === undefined || a === null || a === "") && (b === undefined || b === null || b === ""));
}
// visible: the option means something for this type / outputs / other values
function optVisible(o) {
  const p = state.pipeline;
  if (o.applies_to && !o.applies_to.includes(state.type)) return false;
  if (o.requires_output && !o.requires_output.some((x) => state.outputs.has(x))) return false;
  for (const [dep, want] of Object.entries(o.requires_option || {})) {
    const d = p.options.find((x) => x.id === dep);
    if (d && optValue(d) !== want) return false;
  }
  return true;
}
// why it can't be used right now (still shown, greyed out)
function optBlocked(o) {
  if (o.available === false) return o.disabled_reason || "not available";
  if (o.requires_analysis === "has_grid") {
    if (!state.analysis) return "needs the analysis -- pick a type first";
    if (!state.analysis.tempo?.has_grid) return "this track has no steady beat, so a bar-exact loop isn't possible";
  }
  return null;
}
function setValue(o, v) {
  if (same(v, optDefault(o, state.type))) delete state.values[o.id];
  else state.values[o.id] = v;
  renderOptions();
  renderOutputs();
  renderRunbar();
}

// ------------------------------------------------------------------ step 1: pipeline

function renderPipelines() {
  const root = $("#pipelines");
  fill(root, ...state.caps.pipelines.map((p) =>
    h("button", {
      type: "button", class: "choice", role: "radio",
      "aria-checked": String(state.pipeline?.id === p.id),
      disabled: p.available === false,
      title: p.available === false ? p.disabled_reason : null,
      onclick: () => selectPipeline(p.id),
    },
    h("span", { class: "choice-title" }, p.label),
    h("span", { class: "choice-desc" }, p.available === false ? p.disabled_reason : p.description))));
  $("#step-pipeline").classList.toggle("done", !!state.pipeline);
}

function selectPipeline(id) {
  const p = state.caps.pipelines.find((x) => x.id === id);
  if (!p || state.pipeline?.id === id) return;
  state.pipeline = p;
  state.type = null;
  state.analysis = null;
  state.analysisFor = null;
  state.values = {};
  state.outputs = new Set(p.outputs.filter((o) => o.default && o.available !== false).map((o) => o.id));
  state.formats = new Set(p.default_formats);
  renderAll();
}

// ------------------------------------------------------------------ step 2: upload

function wireUpload() {
  const input = $("#file-input"), dz = $("#dropzone");
  input.addEventListener("change", () => input.files[0] && doUpload(input.files[0]));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) doUpload(f);
  });
  $("#replace-file").addEventListener("click", () => {
    state.upload = null; state.peaks = null; state.analysis = null; state.analysisFor = null;
    input.value = "";
    renderAll();
    input.click();
  });
}

async function doUpload(file) {
  banner.hide();
  if (!/\.wav$/i.test(file.name) && !/wav/.test(file.type)) {
    banner.show(`"${file.name}" doesn't look like a WAV file. Export or record as .wav and try again.`);
    return;
  }
  $("#dropzone").hidden = true;
  $("#upload-progress").hidden = false;
  const bar = $("#upload-bar"), pct = $("#upload-pct");
  try {
    const meta = await uploadFile(file, (f) => {
      bar.style.width = `${(f * 100).toFixed(1)}%`;
      pct.textContent = f < 1 ? `${Math.round(f * 100)}%` : "checking…";
    });
    state.upload = meta;
    state.peaks = null;
    state.analysis = null;
    state.analysisFor = null;
    renderAll();
    loadPeaks();
    maybeAnalyze();
  } catch (err) {
    banner.show(err.message);
    $("#dropzone").hidden = false;
  } finally {
    $("#upload-progress").hidden = true;
    bar.style.width = "0";
  }
}

async function loadPeaks() {
  const id = state.upload?.id;
  if (!id) return;
  try {
    const pk = await api("GET", `/api/uploads/${id}/peaks?points=1000`);
    if (state.upload?.id === id) { state.peaks = pk; renderWave(); }
  } catch (err) { banner.show(`Couldn't draw the waveform: ${err.message}`); }
}

function renderUpload() {
  const u = state.upload;
  $("#dropzone").hidden = !!u;
  $("#file-info").hidden = !u;
  $("#step-upload").classList.toggle("done", !!u);
  if (!u) return;
  $("#file-name").textContent = u.original_name;
  const p = u.probe;
  fill($("#file-facts"), 
    h("span", {}, fmtTime(p.duration)),
    h("span", {}, `${(p.sample_rate / 1000).toFixed(p.sample_rate % 1000 ? 1 : 0)} kHz`),
    p.bits ? h("span", {}, `${p.bits}-bit`) : null,
    h("span", {}, p.channels === 1 ? "mono" : "stereo"),
    h("span", {}, fmtBytes(u.size)));
  renderWave();
}

function renderWave() {
  const pk = state.peaks, path = $("#wave-path"), overlay = $("#wave-overlay"), markers = $("#wave-markers");
  fill(overlay);
  fill(markers);
  if (!pk) { path.setAttribute("d", ""); fill($("#wave-axis")); return; }
  const n = pk.max.length, W = 1000, H = 120, mid = H / 2;
  let peak = 0;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(pk.max[i]), Math.abs(pk.min[i]));
  const k = peak > 0 ? (mid - 4) / peak : 0;
  let d = "";
  for (let i = 0; i < n; i++) {
    const x = ((i + 0.5) / n) * W;
    d += `M${x.toFixed(1)} ${(mid - pk.max[i] * k).toFixed(1)}V${(mid - pk.min[i] * k + 0.5).toFixed(1)}`;
  }
  path.setAttribute("d", d);
  const dur = pk.duration;
  const X = (t) => Math.max(0, Math.min(W, (t / dur) * W));
  const a = state.analysis;
  if (a && state.pipeline?.id === "music" && a.sections) {
    for (const sec of a.sections) {
      overlay.append(s("rect", { class: `sec sec-${sec.label}`, x: X(sec.start), y: 0, width: Math.max(0, X(sec.end) - X(sec.start)), height: H }));
    }
    if (a.main_drop != null) markers.append(s("line", { class: "marker", x1: X(a.main_drop), x2: X(a.main_drop), y1: 0, y2: H }));
  }
  if (a && state.pipeline?.id === "nature" && a.bed) {
    overlay.append(s("rect", { class: "sec sec-bed", x: X(a.bed.start), y: 0, width: Math.max(0, X(a.bed.end) - X(a.bed.start)), height: H }));
  }
  const ticks = 5;
  fill($("#wave-axis"), ...Array.from({ length: ticks + 1 }, (_, i) => h("span", {}, fmtTime((dur * i) / ticks))));
}

// ------------------------------------------------------------------ step 3: type + analysis

function renderTypes() {
  const p = state.pipeline;
  const show = !!p && !!p.type;
  $("#step-type").hidden = !show;
  if (!show) return;
  $("#type-title").textContent = p.type.label;
  $("#types").setAttribute("aria-label", p.type.label);
  fill($("#types"), ...p.types.map((t) =>
    h("button", {
      type: "button", class: "choice", role: "radio", "aria-checked": String(state.type === t.id),
      onclick: () => selectType(t.id),
    },
    h("span", { class: "choice-title" }, t.label, t.untested ? h("span", { class: "badge warn", title: "This recipe hasn't been A/B tested on real recordings yet" }, "untested") : null),
    h("span", { class: "choice-desc" }, t.summary))));
  $("#step-type").classList.toggle("done", !!state.type);
}

function selectType(id) {
  if (state.type === id) return;
  state.type = id;
  // per-type defaults change; keep only the tags the user typed
  const keep = {};
  for (const [k, v] of Object.entries(state.values)) {
    const o = state.pipeline.options.find((x) => x.id === k);
    if (o && o.group === "tags") keep[k] = v;
  }
  state.values = keep;
  state.analysis = null;
  state.analysisFor = null;
  renderAll();
  maybeAnalyze();
}

async function maybeAnalyze() {
  const p = state.pipeline, u = state.upload;
  if (!p || !u || !p.analyze || (p.type && !state.type)) { renderAnalysis(); return; }
  const key = `${u.id}|${p.id}|${state.type}`;
  if (state.analysisFor === key) return;
  state.analysisFor = key;
  state.analysis = null;
  renderAnalysis(true);
  try {
    const res = await api("POST", `/api/uploads/${u.id}/analyze`, { pipeline: p.id, type: state.type || "" });
    if (state.analysisFor !== key) return;
    state.analysis = res.analysis;
  } catch (err) {
    if (state.analysisFor !== key) return;
    state.analysisFor = null;
    banner.show(`Analysis failed: ${err.message}`);
  }
  renderAll();
}

function stat(label, value) {
  return h("div", { class: "stat" }, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value));
}

function renderAnalysis(loading = false) {
  const box = $("#analysis"), p = state.pipeline;
  const relevant = p && p.analyze && state.upload && (!p.type || state.type);
  box.hidden = !relevant;
  if (!relevant) return;
  if (loading || !state.analysis) {
    fill(box, h("div", { class: "spinner-row" }, h("span", { class: "spinner" }),
      p.id === "music" ? "Finding the tempo, key, sections and drop…" : "Finding the clean bed…"));
    return;
  }
  const a = state.analysis;
  if (p.id === "music") {
    const t = a.tempo || {}, k = a.key || {}, l = a.loudness || {};
    const labels = [...new Set((a.sections || []).map((x) => x.label))];
    fill(box, 
      h("div", { class: "stats" },
        stat("Tempo", t.has_grid ? `${fmtNum(t.bpm, 0.01)} BPM` : "no steady beat"),
        stat("Key", k.name ? `${k.name} · ${k.camelot}` : "–"),
        stat("Loudness", l.integrated_lufs != null ? `${fmtNum(l.integrated_lufs, 0.1)} LUFS` : "–"),
        stat("True peak", l.true_peak_dbtp != null ? `${fmtNum(l.true_peak_dbtp, 0.1)} dBTP` : "–"),
        stat("Main drop", a.main_drop != null ? fmtTime(a.main_drop) : "–")),
      labels.length ? h("div", { class: "legend" }, ...labels.map((lb) =>
        h("span", {}, h("i", { class: `sw-${lb}` }), lb))) : null,
      (a.flags || []).length ? h("ul", { class: "flags" }, ...a.flags.map((f) => h("li", {}, f))) : null);
  } else if (p.id === "nature") {
    const b = a.bed || {};
    fill(box, 
      h("div", { class: "stats" },
        stat("Clean bed", `${fmtTime(b.start)} – ${fmtTime(b.end)}`),
        stat("Clean runs", String(a.clean_runs ?? "–")),
        stat("Short looping", a.short_loop_mode || "–")),
      h("p", { class: "muted small" }, b.why ? `Bed choice: ${b.why}. Shaded green on the waveform.` : ""));
  }
  renderWave();
}

// ------------------------------------------------------------------ step 4: options

function optionControl(o, blocked) {
  const v = optValue(o);
  const id = `opt-${o.id}`;
  switch (o.kind) {
    case "bool":
      return h("label", { class: "switch" },
        h("input", { type: "checkbox", role: "switch", id, checked: !!v, disabled: !!blocked,
          onchange: (e) => setValue(o, e.target.checked) }),
        h("span", { class: "muted small" }, v ? "on" : "off"));
    case "number": {
      if (v === undefined || v === null) {
        // no default (e.g. a custom loudness that overrides the preset):
        // nothing is sent until the user opts in
        const start = o.suggest ?? Math.round(((o.min + o.max) / 2) / (o.step || 1)) * (o.step || 1);
        return h("div", { class: "range-row" },
          h("span", { class: "muted small" }, "Not set"),
          h("button", { type: "button", class: "btn small", id, disabled: !!blocked, onclick: () => setValue(o, start) }, "Customize"));
      }
      const out = h("output", { for: id }, `${fmtNum(v, o.step)}${o.unit ? " " + o.unit : ""}`);
      const range = h("input", {
        type: "range", id, min: o.min, max: o.max, step: o.step ?? "any",
        value: v ?? o.min, disabled: !!blocked,
        oninput: (e) => { out.textContent = `${fmtNum(+e.target.value, o.step)}${o.unit ? " " + o.unit : ""}`; },
        onchange: (e) => setValue(o, +e.target.value),
      });
      return h("div", { class: "range-row" }, range, out);
    }
    case "choice":
      if (o.choices.length <= 5) {
        return h("div", { class: "seg", role: "group", "aria-labelledby": `${id}-label` },
          ...o.choices.map((c) => h("button", {
            type: "button", "aria-pressed": String(String(v) === c.value), disabled: !!blocked,
            onclick: () => setValue(o, c.value),
          }, c.label)));
      }
      return h("select", { id, disabled: !!blocked, onchange: (e) => setValue(o, e.target.value) },
        ...o.choices.map((c) => h("option", { value: c.value, selected: String(v) === c.value }, c.label)));
    case "time":
      return h("input", {
        type: "text", id, inputmode: "decimal", placeholder: "e.g. 90 or 1:30", value: v ?? "",
        pattern: "^(\\d+(\\.\\d+)?|\\d+:\\d{1,2}(\\.\\d+)?|\\d+:\\d{1,2}:\\d{1,2}(\\.\\d+)?)$",
        disabled: !!blocked, autocomplete: "off",
        onchange: (e) => { const t = e.target.value.trim(); if (!t || e.target.checkValidity()) setValue(o, t); },
      });
    case "time_list":
      return h("input", {
        type: "text", id, placeholder: "e.g. 0:15, 2:40, end", value: (v || []).join(", "),
        disabled: !!blocked, autocomplete: "off",
        onchange: (e) => setValue(o, e.target.value.split(",").map((x) => x.trim()).filter(Boolean)),
      });
    case "text": {
      const ph = o.id === "title" && state.upload ? state.upload.original_name.replace(/\.[^.]+$/, "") : "";
      return h("input", {
        type: "text", id, value: v ?? "", maxlength: 200, placeholder: ph, disabled: !!blocked, autocomplete: "off",
        onchange: (e) => setValue(o, e.target.value),
      });
    }
  }
  return h("span", { class: "muted" }, `(${o.kind})`);
}

function optionCard(o) {
  const blocked = optBlocked(o);
  const changed = o.id in state.values;
  const def = optDefault(o, state.type);
  const detail = o.detail_by_type?.[state.type] ?? o.detail;
  let defText = null;
  if (changed && def !== undefined && def !== null && def !== "") {
    defText = o.kind === "bool" ? (def ? "on" : "off")
      : o.kind === "choice" ? (o.choices.find((c) => c.value === String(def))?.label ?? def)
        : o.kind === "number" ? `${fmtNum(def, o.step)}${o.unit ? " " + o.unit : ""}` : String(def);
  }
  return h("div", { class: `opt${changed ? " changed" : ""}${blocked ? " disabled" : ""}` },
    h("div", { class: "opt-top" },
      h("label", { class: "opt-label", for: `opt-${o.id}`, id: `opt-${o.id}-label` }, o.label),
      changed ? h("button", { type: "button", class: "reset", title: defText ? `Back to ${defText}` : "Back to default",
        onclick: () => { delete state.values[o.id]; renderOptions(); renderOutputs(); renderRunbar(); } }, "reset") : null),
    optionControl(o, blocked),
    blocked ? h("div", { class: "opt-reason" }, blocked) : null,
    o.help ? h("div", { class: "opt-hint" }, o.help) : null,
    detail ? h("div", { class: "opt-detail" }, detail) : null);
}

function groupsFor(advanced) {
  const p = state.pipeline;
  const out = [];
  for (const g of p.groups) {
    const opts = p.options.filter((o) => !!o.advanced === advanced && o.group === g.id && optVisible(o));
    if (opts.length) out.push(h("div", { class: "opt-group" }, h("h3", {}, g.label), h("div", { class: "opt-grid" }, ...opts.map(optionCard))));
  }
  return out;
}

function renderOptions() {
  const p = state.pipeline;
  const ready = p && state.upload && (!p.type || state.type);
  $("#step-options").hidden = !ready;
  if (!ready) return;
  $("#options-num").textContent = p.type ? "4" : "3";
  // keep the Advanced disclosure's open/closed state; it starts collapsed
  fill($("#options-basic"), ...groupsFor(false));
  const adv = groupsFor(true);
  fill($("#options-advanced"), ...adv);
  $("#advanced").hidden = adv.length === 0;
  const nAdv = p.options.filter((o) => o.advanced && optVisible(o)).length;
  const nChanged = p.options.filter((o) => o.advanced && o.id in state.values).length;
  $("#advanced-count").textContent = `${nAdv} settings${nChanged ? ` · ${nChanged} changed` : ""}`;
  $("#reset-options").hidden = Object.keys(state.values).length === 0;
}

// ------------------------------------------------------------------ step 5: outputs + formats

function renderOutputs() {
  const p = state.pipeline;
  const ready = p && state.upload && (!p.type || state.type);
  $("#step-outputs").hidden = !ready;
  if (!ready) return;
  $("#outputs-num").textContent = p.type ? "5" : "4";
  fill($("#outputs"), ...p.outputs.map((o) => h("label", { class: "chip", title: o.disabled_reason || null },
    h("input", { type: "checkbox", checked: state.outputs.has(o.id), disabled: o.available === false,
      onchange: (e) => {
        e.target.checked ? state.outputs.add(o.id) : state.outputs.delete(o.id);
        renderOptions(); renderOutputs(); renderRunbar();
      } }),
    h("span", {}, o.label, o.slow ? h("small", {}, "slow") : null))));
  const needFormats = p.outputs.some((o) => o.formats && state.outputs.has(o.id));
  $("#formats-wrap").hidden = !needFormats;
  fill($("#formats"), ...state.caps.formats.map((f) => h("label", { class: "chip", title: f.disabled_reason || null },
    h("input", { type: "checkbox", checked: state.formats.has(f.id), disabled: f.available === false,
      onchange: (e) => { e.target.checked ? state.formats.add(f.id) : state.formats.delete(f.id); renderRunbar(); } }),
    h("span", {}, f.label))));
}

// ------------------------------------------------------------------ run bar, plan, run

function jobRequest() {
  const p = state.pipeline;
  const values = { ...state.values };
  // the server stores every upload as input.wav, so an empty title would
  // become "input" -- default it to the uploaded file's own name
  if (p.options.some((o) => o.id === "title") && !values.title) {
    values.title = state.upload.original_name.replace(/\.[^.]+$/, "").slice(0, 200);
  }
  return {
    upload_id: state.upload.id,
    pipeline: p.id,
    type: state.type || "",
    values,
    outputs: p.outputs.filter((o) => state.outputs.has(o.id)).map((o) => o.id),
    formats: state.caps.formats.filter((f) => state.formats.has(f.id)).map((f) => f.id),
  };
}

function runProblem() {
  const p = state.pipeline;
  if (!p) return "Pick what kind of audio this is.";
  if (!state.upload) return "Upload a WAV.";
  if (p.type && !state.type) return `Pick: ${p.type.label.toLowerCase()}`;
  if (state.outputs.size === 0) return "Choose at least one output.";
  const needFormats = p.outputs.some((o) => o.formats && state.outputs.has(o.id));
  if (needFormats && state.formats.size === 0) return "Choose at least one audio format.";
  return null;
}

function renderRunbar() {
  const p = state.pipeline;
  $("#runbar").hidden = !p || !state.upload;
  if ($("#runbar").hidden) return;
  const problem = runProblem();
  $("#run-btn").disabled = !!problem;
  $("#plan-btn").disabled = !!problem;
  if (problem) { $("#run-summary").textContent = problem; return; }
  const outs = p.outputs.filter((o) => state.outputs.has(o.id)).map((o) => o.label);
  const typeLabel = p.types.find((t) => t.id === state.type)?.label;
  const nChanged = Object.keys(state.values).length;
  $("#run-summary").textContent = [
    [p.label, typeLabel].filter(Boolean).join(" · "),
    outs.join(", "),
    nChanged ? `${nChanged} setting${nChanged > 1 ? "s" : ""} changed` : "recipe defaults",
  ].join(" — ");
}

async function showPlan() {
  banner.hide();
  const btn = $("#plan-btn");
  btn.disabled = true;
  btn.textContent = "Planning…";
  try {
    const plan = await api("POST", "/api/plan", jobRequest());
    renderPlan(plan);
    $("#plan-dialog").showModal();
  } catch (err) {
    banner.show(err.message);
  } finally {
    btn.textContent = "Show plan";
    renderRunbar();
  }
}

function renderPlan(plan) {
  const body = $("#plan-body");
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  const facts = [];
  const add = (k, v) => { if (v !== undefined && v !== null && v !== "" && !(Array.isArray(v) && !v.length)) facts.push([k, v]); };
  add("Pipeline", plan.pipeline);
  add("Type", plan.genre || plan.label);
  add("Outputs", (plan.outputs || []).join(", "));
  add("Formats", (plan.formats || []).join(", "));
  add("Visual style", plan.style && `${plan.style} ${(plan.visual_args || []).join(" ")}`.trim());
  if (plan.bed) add("Bed", `${fmtTime(plan.bed.start)} – ${fmtTime(plan.bed.end)} (${plan.bed.why})`);
  if (plan.targets) add("Targets", JSON.stringify(plan.targets));
  add("Preset", plan.preset);
  add("Short looping", plan.short_loop_mode);
  if (plan.untested_recipe) add("Note", "this recipe is marked untested -- A/B it before trusting it");
  fill(body, 
    steps.length ? h("div", {}, h("h3", { class: "subhead" }, "Steps"),
      h("ol", { class: "plan-steps" }, ...steps.map((st) => h("li", {}, st.label || st.step)))) : null,
    facts.length ? h("dl", { class: "kv" }, ...facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))])) : null,
    plan.eq_chain || plan.chain ? h("details", { class: "mono-details" }, h("summary", {}, "Filter chain"),
      h("pre", { class: "mono" }, plan.eq_chain || plan.chain)) : null,
    h("details", { class: "mono-details" }, h("summary", {}, "Full plan (JSON)"), h("pre", { class: "mono" }, JSON.stringify(plan, null, 2))));
}

async function run() {
  banner.hide();
  $("#plan-dialog").close();
  const btn = $("#run-btn");
  btn.disabled = true;
  try {
    const job = await api("POST", "/api/jobs", jobRequest());
    location.hash = `#/job/${job.id}`;
    refreshJobCount();
  } catch (err) {
    banner.show(err.message);
    renderRunbar();
  }
}

// ------------------------------------------------------------------ job view

let es = null;          // EventSource for the open job
let currentJob = null;
let tick = null;

function closeStream() {
  if (es) { es.close(); es = null; }
  if (tick) { clearInterval(tick); tick = null; }
}

async function openJob(id) {
  closeStream();
  $("#wizard").hidden = true;
  $("#jobview").hidden = false;
  $("#job-log").textContent = "";
  $("#results").hidden = true;
  fill($("#results"));
  currentJob = null;
  try {
    const job = await api("GET", `/api/jobs/${id}`);
    renderJob(job);
    if (job.state === "queued" || job.state === "running") follow(id);
  } catch (err) {
    banner.show(err.message);
    location.hash = "#/";
  }
}

function follow(id) {
  es = new EventSource(`/api/jobs/${id}/events`);
  const log = $("#job-log");
  es.addEventListener("log", (e) => {
    const msg = JSON.parse(e.data);
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.append(msg.line + "\n");
    if (log.childNodes.length > 1200) fill(log, log.textContent.split("\n").slice(-600).join("\n"));
    if (atBottom) log.scrollTop = log.scrollHeight;
  });
  es.addEventListener("job", (e) => {
    const job = JSON.parse(e.data).job;
    renderJob(job);
    if (!["queued", "running"].includes(job.state)) { closeStream(); refreshJobCount(); }
  });
  es.onerror = () => {
    // the server closes the stream when a job ends; anything else, poll once
    if (currentJob && ["queued", "running"].includes(currentJob.state)) {
      setTimeout(() => currentJob && api("GET", `/api/jobs/${currentJob.id}`).then(renderJob).catch(() => {}), 2000);
    }
  };
  tick = setInterval(() => currentJob && renderJobSub(currentJob), 1000);
}

const PIPE_LABEL = (id) => state.caps?.pipelines.find((p) => p.id === id)?.label || id;
const TYPE_LABEL = (pid, tid) => state.caps?.pipelines.find((p) => p.id === pid)?.types.find((t) => t.id === tid)?.label || tid;

function renderJobSub(job) {
  const r = job.request;
  const started = job.started_at ? new Date(job.started_at) : null;
  const ended = job.finished_at ? new Date(job.finished_at) : new Date();
  const took = started ? fmtTime((ended - started) / 1000) : null;
  $("#job-sub").textContent = [PIPE_LABEL(r.pipeline), r.type && TYPE_LABEL(r.pipeline, r.type),
    job.state === "queued" ? "waiting for a free worker" : took && (job.state === "running" ? `running ${took}` : `took ${took}`)]
    .filter(Boolean).join(" · ");
}

function renderJob(job) {
  const prev = currentJob;
  currentJob = job;
  $("#job-title").textContent = job.upload_name;
  renderJobSub(job);
  const st = $("#job-state");
  st.textContent = job.state;
  st.className = `state ${job.state}`;
  $("#job-error").hidden = !job.error || job.state === "succeeded";
  $("#job-error").textContent = job.error || "";
  $("#job-command").textContent = job.command;
  fill($("#job-steps"), ...(job.steps || []).map((s) => {
    const pct = s.total ? Math.min(100, (100 * (s.done || 0)) / s.total) : null;
    const meta = s.status === "running"
      ? [s.detail, pct != null ? `${Math.round(pct)}%` : null].filter(Boolean).join(" · ")
      : s.status === "failed" ? (s.detail || "failed") : s.status === "done" ? "done" : "";
    const li = h("li", { class: s.status },
      h("span", { class: "dot", "aria-hidden": "true" }),
      h("span", { class: "step-label" }, s.label),
      h("span", { class: "step-meta" }, meta));
    if (s.status === "running") {
      const bar = h("div", { class: `bar${pct == null ? " indeterminate" : ""}` }, h("div", { class: "bar-fill" }));
      if (pct != null) bar.firstChild.style.width = `${pct}%`;
      li.append(bar);
    }
    return li;
  }));
  const live = job.state === "queued" || job.state === "running";
  $("#cancel-btn").hidden = !live;
  $("#rerun-btn").hidden = live;
  $("#zip-btn").hidden = !(job.files || []).length;
  $("#zip-btn").href = `/api/jobs/${job.id}/zip`;
  if (job.state === "succeeded" && (!prev || prev.state !== "succeeded" || prev.id !== job.id)) renderResults(job);
  if (!live && !$("#job-log").textContent) loadLogTail(job);
}

async function loadLogTail(job) {
  // finished jobs: the log isn't streamed; fetch the snapshot stream once
  // (it replays the recent lines, then closes)
  try {
    const res = await fetch(`/api/jobs/${job.id}/events`);
    const text = await res.text();
    const lines = text.split("\n").filter((l) => l.startsWith("data: ")).map((l) => JSON.parse(l.slice(6)))
      .filter((m) => m.type === "log").map((m) => m.line);
    $("#job-log").textContent = lines.join("\n");
  } catch { /* the log is a nicety */ }
}

function fileURL(job, f, inline) { return `/api/jobs/${job.id}/files/${f.index}${inline ? "?inline=1" : ""}`; }

const KIND_LABEL = {
  master: "Master", master_long: "Long master", master_short: "Short master (3 min)", clip: "Short/Reel clip",
  video_16x9: "YouTube video (16:9)", video_9x16: "Short / Reel video (9:16)", video_1x1: "Instagram feed video (1:1)",
  video_long: "Long video", video_short: "Short video", captions: "Captions", analysis: "Analysis", report: "Report", thumbnail: "Thumbnail",
};
const FORMAT_LABEL = { wav24: "WAV 24-bit", wav16: "WAV 16-bit", flac: "FLAC", mp3: "MP3", wav: "WAV", mp4: "MP4", txt: "Text", json: "JSON", png: "PNG", jpg: "JPEG" };

function resultItem(job, f) {
  const url = fileURL(job, f, true);
  const top = h("div", { class: "result-top" },
    h("span", { class: "result-name" }, `${KIND_LABEL[f.kind] || f.kind}`),
    h("span", { class: "badge" }, FORMAT_LABEL[f.format] || f.format),
    h("span", { class: "muted small tabular" }, fmtBytes(f.size)),
    h("a", { class: "btn small", href: fileURL(job, f, false), download: f.name }, "Download"));
  const item = h("div", { class: "result" }, top);
  if (["wav24", "wav16", "wav", "mp3", "flac"].includes(f.format)) {
    item.append(h("audio", { controls: true, preload: "none", src: url }));
  } else if (f.format === "mp4") {
    const v = h("video", { controls: true, preload: "metadata", src: url, playsinline: true });
    if (f.kind === "video_9x16") v.classList.add("portrait");
    item.append(v);
  } else if (f.format === "png" || f.format === "jpg") {
    item.append(h("img", { src: url, alt: KIND_LABEL[f.kind] || f.name, loading: "lazy" }));
  } else if (f.kind === "captions") {
    const pre = h("pre", {}, "loading…");
    fetch(url).then((r) => r.text()).then((t) => { pre.textContent = t; });
    top.insertBefore(h("button", { type: "button", class: "btn small", onclick: async (e) => {
      await navigator.clipboard.writeText(pre.textContent);
      e.target.textContent = "Copied";
      setTimeout(() => { e.target.textContent = "Copy"; }, 1500);
    } }, "Copy"), top.lastChild);
    item.append(pre);
  } else if (f.format === "txt") {
    const det = h("details", { class: "mono-details" }, h("summary", {}, "Show"));
    det.addEventListener("toggle", () => {
      if (det.open && det.childNodes.length === 1) fetch(url).then((r) => r.text()).then((t) => det.append(h("pre", { class: "mono" }, t)));
    }, { once: false });
    item.append(det);
  }
  return item;
}

function renderResults(job) {
  const files = job.files || [];
  const groups = [
    ["Audio", files.filter((f) => ["wav24", "wav16", "wav", "flac", "mp3"].includes(f.format))],
    ["Video", files.filter((f) => f.format === "mp4")],
    ["Captions & analysis", files.filter((f) => ["captions", "analysis", "thumbnail"].includes(f.kind))],
    ["Report", files.filter((f) => f.kind === "report")],
  ];
  fill($("#results"), ...groups.filter(([, fs]) => fs.length).map(([title, fs]) =>
    h("section", { class: "card result-group" }, h("h2", {}, title), h("div", { class: "result-list" }, ...fs.map((f) => resultItem(job, f))))));
  $("#results").hidden = files.length === 0;
}

async function cancelJob() {
  if (!currentJob) return;
  try { renderJob(await api("POST", `/api/jobs/${currentJob.id}/cancel`)); } catch (err) { banner.show(err.message); }
}

// "Edit & run again": load the job's request back into the form
async function rerun() {
  const job = currentJob;
  if (!job) return;
  try {
    const up = await api("GET", `/api/uploads/${job.upload_id}`);
    const p = state.caps.pipelines.find((x) => x.id === job.request.pipeline);
    if (!p) throw new Error("that pipeline no longer exists");
    state.pipeline = p;
    state.upload = up;
    state.type = job.request.type || null;
    state.values = { ...(job.request.values || {}) };
    for (const k of Object.keys(state.values)) if (!p.options.some((o) => o.id === k)) delete state.values[k];
    state.outputs = new Set(job.request.outputs || []);
    state.formats = new Set(job.request.formats || p.default_formats);
    state.peaks = null;
    state.analysis = null;
    state.analysisFor = null;
    location.hash = "#/";
    loadPeaks();
    maybeAnalyze();
  } catch (err) {
    banner.show(`Can't re-run: ${err.message} (the upload may have been deleted -- upload the file again)`);
  }
}

// ------------------------------------------------------------------ jobs drawer

async function refreshJobs() {
  try {
    const { jobs } = await api("GET", "/api/jobs");
    const active = jobs.filter((j) => j.state === "queued" || j.state === "running").length;
    $("#jobs-active").hidden = active === 0;
    $("#jobs-active").textContent = String(active);
    if ($("#jobs-drawer").hidden) return;
    fill($("#job-list"), ...(jobs.length ? jobs.map((j) => h("li", { class: "job-item" },
      h("a", { href: `#/job/${j.id}`, onclick: () => toggleDrawer(false) }, j.upload_name),
      h("span", { class: `state ${j.state}` }, j.state),
      h("span", { class: "muted" }, [PIPE_LABEL(j.request.pipeline), j.request.type && TYPE_LABEL(j.request.pipeline, j.request.type), ago(j.created_at)].filter(Boolean).join(" · ")),
      ["queued", "running"].includes(j.state) ? null : h("button", { type: "button", class: "btn ghost small", title: "Delete this job and its files",
        onclick: async () => {
          if (!confirm(`Delete the job for "${j.upload_name}" and its output files?`)) return;
          try { await api("DELETE", `/api/jobs/${j.id}`); if (currentJob?.id === j.id) location.hash = "#/"; refreshJobs(); } catch (err) { banner.show(err.message); }
        } }, "Delete"))) : [h("li", { class: "empty" }, "No jobs yet.")]));
  } catch { /* the badge is a nicety */ }
}
const refreshJobCount = refreshJobs;

let drawerTimer = null;
function toggleDrawer(open) {
  const d = $("#jobs-drawer");
  open = open ?? d.hidden;
  d.hidden = !open;
  $("#jobs-toggle").setAttribute("aria-expanded", String(open));
  clearInterval(drawerTimer);
  if (open) { refreshJobs(); drawerTimer = setInterval(refreshJobs, 3000); }
}

// ------------------------------------------------------------------ theme

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("pa-theme"); } catch { /* private mode */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pa-theme", next); } catch { /* */ }
  });
}

// ------------------------------------------------------------------ routing + boot

function renderAll() {
  renderPipelines();
  renderUpload();
  renderTypes();
  renderAnalysis();
  renderOptions();
  renderOutputs();
  renderRunbar();
}

function route() {
  if (!$("#jobs-drawer").hidden) toggleDrawer(false);
  const m = location.hash.match(/^#\/job\/([0-9a-f]{20})$/);
  if (m) { openJob(m[1]); return; }
  closeStream();
  currentJob = null;
  $("#jobview").hidden = true;
  $("#wizard").hidden = false;
  renderAll();
}

async function boot() {
  initTheme();
  wireUpload();
  $("#banner-close").addEventListener("click", banner.hide);
  $("#jobs-toggle").addEventListener("click", () => toggleDrawer());
  $("#jobs-close").addEventListener("click", () => toggleDrawer(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#jobs-drawer").hidden) toggleDrawer(false); });
  $("#plan-btn").addEventListener("click", showPlan);
  $("#run-btn").addEventListener("click", run);
  $("#plan-run").addEventListener("click", run);
  $("#plan-close").addEventListener("click", () => $("#plan-dialog").close());
  $("#plan-cancel").addEventListener("click", () => $("#plan-dialog").close());
  $("#cancel-btn").addEventListener("click", cancelJob);
  $("#rerun-btn").addEventListener("click", rerun);
  $("#reset-options").addEventListener("click", () => { state.values = {}; renderAll(); });
  window.addEventListener("hashchange", route);
  try {
    state.caps = await api("GET", "/api/capabilities");
  } catch (err) {
    banner.show(`Couldn't load the pipelines: ${err.message}`);
    return;
  }
  refreshJobs();
  setInterval(refreshJobs, 10000);
  route();
}

boot();
