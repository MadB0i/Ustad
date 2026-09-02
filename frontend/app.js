/* Ustad frontend.
 *
 * Every animated pixel here is downstream of a real backend event. Packets in the
 * channel are spawned by `dataset.pair` and `train.step`; the brightness of a layer row
 * is that layer's actual LoRA gradient norm from `train.layer_grads`; the sparkline
 * plots the loss values the optimiser produced. When nothing is running, nothing moves —
 * there is no idle animation and no synthetic data anywhere in this file.
 */

'use strict';

const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const COLOR = {
  teacher: [224, 160, 80],
  student: [84, 199, 192],
  line: '#2c3040',
  faint: '#6d7484',
  muted: '#9aa0b0',
  text: '#e6e7ec',
  panel: '#191c25',
};

const $ = (id) => document.getElementById(id);

const state = {
  system: null,
  teachers: [],
  resident: [],
  teacher: '',
  datasets: [],
  runs: [],
  job: null,

  build: { active: false, phase: '', prompts: 0, pairs: 0, target: 0, rejected: 0 },
  streams: new Map(),
  generating: 0,

  train: {
    active: false, step: 0, total: 0, layers: 0, loraLayers: [],
    losses: [], emas: [], evals: [], norms: [], normPeak: 1e-9, last: null,
  },

  packets: [],
  packetTotal: 0,
  sweeps: [],

  page: { name: '', offset: 0, limit: 40, total: 0 },
};

/* ─── http ──────────────────────────────────────────────────────────────── */

async function api(path, options) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text }; }
  if (!response.ok) throw new Error((body && body.detail) || `${response.status} ${response.statusText}`);
  return body;
}

/* ─── formatting ────────────────────────────────────────────────────────── */

const bytes = (n) => {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i >= 2 ? 2 : 0)} ${units[i]}`;
};

const duration = (s) => {
  if (s == null) return '—';
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function mix(a, b, t) {
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}

function rgba(c, alpha) {
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha.toFixed(3)})`;
}

/* ─── chips ─────────────────────────────────────────────────────────────── */

function setChip(id, cls, text) {
  const el = $(id);
  el.className = `chip ${cls}`;
  el.innerHTML = `<i></i>${text}`;
  el.title = text;
}

/* ─── boot ──────────────────────────────────────────────────────────────── */

async function loadSystem() {
  let system;
  try {
    system = await api('/api/system');
  } catch (err) {
    setChip('chip-gpu', 'bad', `backend unreachable: ${err.message}`);
    return;
  }
  state.system = system;
  const hw = system.hardware;

  if (hw.cuda_available && hw.arch_supported) {
    setChip('chip-gpu', 'ok',
      `${hw.device_name} · ${bytes(hw.vram_total_bytes)} · sm_${(hw.capability || '').replace('.', '')} · ${system.tier}`);
  } else if (hw.cuda_available) {
    setChip('chip-gpu', 'bad', `${hw.device_name}: this torch build has no kernels for it — CPU only`);
  } else {
    setChip('chip-gpu', 'warn', `CPU only · ${hw.cpu_threads} threads · torch ${hw.torch || '?'}`);
  }

  setChip('chip-ollama', system.ollama.ok ? 'ok' : 'bad',
    system.ollama.ok ? `Ollama ${system.ollama.version}` : (system.ollama.error || 'Ollama unreachable'));

  const select = $('student-select');
  select.innerHTML = '';
  for (const model of system.students) {
    const option = document.createElement('option');
    option.value = model.repo_id;
    option.textContent = `${model.label} · ${model.params}${model.cached ? '' : ' · not downloaded'}`;
    option.dataset.note = model.note;
    option.dataset.cached = model.cached ? '1' : '';
    select.appendChild(option);
  }
  select.value = system.default_student;

  const preset = system.preset;
  $('tr-epochs').value = preset.epochs;
  $('tr-seq').value = preset.max_seq_len;
  $('tr-rank').value = preset.lora_r;
  $('tr-alpha').value = preset.lora_alpha;
  $('tr-lr').value = preset.lr;
  $('tr-accum').value = preset.grad_accum;
  $('tr-4bit').checked = preset.load_in_4bit;
  $('tr-ckpt').checked = preset.gradient_checkpointing;

  applyJobState(system.jobs);
  onStudentChange();

  for (const note of hw.notes || []) log('system', note);
}

async function loadTeachers() {
  let payload;
  try {
    payload = await api('/api/teachers');
  } catch (err) {
    $('teacher-hint').textContent = err.message;
    return;
  }
  state.teachers = payload.models;
  state.resident = payload.resident || [];
  setChip('chip-ollama', payload.ollama.ok ? 'ok' : 'bad',
    payload.ollama.ok ? `Ollama ${payload.ollama.version}` : (payload.ollama.error || 'Ollama unreachable'));
  renderTeachers();
}

function renderTeachers() {
  const host = $('teacher-list');
  host.innerHTML = '';
  const residentNames = new Set(state.resident.map((r) => r.model || r.name));

  if (!state.teachers.length) {
    const hints = (state.system?.recommended_teachers || [])
      .map((t) => `ollama pull ${t.tag}  (${t.size_gb} GB — ${t.note})`).join('\n');
    host.innerHTML = `<div class="empty-state">No models pulled yet. In a terminal:\n\n${hints}</div>`;
    $('teacher-hint').textContent = 'Ustad only uses models Ollama already has — it never downloads teachers itself.';
    return;
  }

  for (const model of state.teachers) {
    const usable = model.chat_capable !== false;
    const row = document.createElement('div');
    row.className = 'model' + (state.teacher === model.name ? ' selected' : '') + (usable ? '' : ' unusable');
    row.dataset.name = model.name;
    row.setAttribute('role', 'radio');
    row.setAttribute('aria-checked', String(state.teacher === model.name));

    const badges = [];
    if (model.fits_gpu === true) badges.push('<span class="badge fit">fits VRAM</span>');
    if (model.fits_gpu === false) badges.push('<span class="badge spill">spills to CPU</span>');
    if (residentNames.has(model.name)) badges.push('<span class="badge resident">loaded</span>');
    if (!usable) badges.push('<span class="badge">no chat template</span>');
    if (model.thinking) badges.push('<span class="badge">thinking</span>');

    row.innerHTML =
      `<span class="model-name">${escapeHtml(model.name)}</span>` +
      `<span class="model-meta">${escapeHtml(model.parameter_size || '?')} · ${escapeHtml(model.quantization || '?')} · ${bytes(model.size_bytes)}</span>` +
      `<span class="model-badges">${badges.join(' ')}</span>`;

    if (usable) row.addEventListener('click', () => selectTeacher(model.name));
    host.appendChild(row);
  }

  if (!state.teacher) {
    const first = state.teachers.find((m) => m.chat_capable !== false);
    if (first) selectTeacher(first.name);
  }
}

function selectTeacher(name) {
  state.teacher = name;
  const model = state.teachers.find((m) => m.name === name);
  $('teacher-note').textContent = name;
  $('teacher-hint').textContent = model
    ? `${model.parameter_size || '?'} parameters, ${model.quantization || '?'}${model.fits_gpu === false ? ' — larger than VRAM, generation will be slow' : ''}`
    : '';
  renderTeachers();
}

async function loadDatasets() {
  const { datasets } = await api('/api/datasets');
  state.datasets = datasets;
  for (const id of ['tr-dataset', 'data-select']) {
    const select = $(id);
    const previous = select.value;
    select.innerHTML = '';
    for (const set of datasets) {
      const option = document.createElement('option');
      option.value = set.name;
      option.textContent = `${set.name} · ${set.rows} rows`;
      select.appendChild(option);
    }
    if (datasets.some((d) => d.name === previous)) select.value = previous;
  }
  $('drawer-meta').textContent =
    `${datasets.length} datasets · ${datasets.reduce((n, d) => n + d.rows, 0)} rows · ${state.runs.length} runs`;
  if (datasets.length && !state.page.name) loadPage(datasets[0].name, 0);
}

async function loadRuns() {
  const { runs } = await api('/api/runs');
  state.runs = runs;

  const select = $('export-run');
  const previous = select.value;
  select.innerHTML = '';
  for (const run of runs.filter((r) => r.has_adapter)) {
    const option = document.createElement('option');
    option.value = run.run_id;
    option.textContent = `${run.run_id}${run.final_loss != null ? ` · loss ${run.final_loss}` : ''}`;
    select.appendChild(option);
  }
  if (previous) select.value = previous;

  const host = $('runs-list');
  host.innerHTML = '';
  if (!runs.length) {
    host.innerHTML = '<div class="empty-state">No runs yet.</div>';
    return;
  }
  for (const run of runs) {
    const row = document.createElement('div');
    row.className = 'row wide';
    const kv = [
      ['student', run.student || '?'],
      ['dataset', run.dataset || '?'],
      ['steps', `${run.steps_completed ?? '?'}/${run.total_steps ?? '?'}`],
      ['loss', run.first_loss != null ? `${run.first_loss} → ${run.final_loss}` : '?'],
      ['eval', run.best_eval_loss ?? '—'],
      ['peak vram', run.peak_vram ? bytes(run.peak_vram) : '—'],
      ['elapsed', duration(run.elapsed)],
    ].map(([k, v]) => `<span>${k} <b>${escapeHtml(v)}</b></span>`).join('');
    row.innerHTML =
      `<div><div class="title">${escapeHtml(run.run_id)}${run.stopped_early ? ' · stopped early' : ''}</div><div class="kv">${kv}</div></div>` +
      `<div><button class="ghost small" data-export="${escapeHtml(run.run_id)}"${run.has_adapter ? '' : ' disabled'}>Export</button></div>`;
    host.appendChild(row);
  }
  host.querySelectorAll('[data-export]').forEach((button) => {
    button.addEventListener('click', () => {
      $('export-run').value = button.dataset.export;
      $('export-name').value = `${button.dataset.export}-export`;
      switchTab('export');
    });
  });
}

async function loadExports() {
  const { exports } = await api('/api/exports');
  const host = $('export-list');
  host.innerHTML = '';
  if (!exports.length) {
    host.innerHTML = '<div class="empty-state">Nothing exported yet. Pick a run above.</div>';
    return;
  }
  for (const item of exports) {
    const row = document.createElement('div');
    row.className = 'row wide';
    row.innerHTML =
      `<div><div class="title">${escapeHtml(item.name)} · ${bytes(item.size_bytes)}${item.merged ? ' · merged' : ' · adapter only'}</div>` +
      `<div class="kv"><span>path <b>${escapeHtml(item.path)}</b></span></div>` +
      (item.ollama_command ? `<pre>${escapeHtml(item.ollama_command)}</pre>` : '') +
      (item.gguf_command ? `<pre>${escapeHtml(item.gguf_command)}</pre>` : '') +
      `</div><div></div>`;
    host.appendChild(row);
  }
}

/* ─── dataset preview ───────────────────────────────────────────────────── */

async function loadPage(name, offset) {
  if (!name) return;
  let page;
  try {
    page = await api(`/api/datasets/${encodeURIComponent(name)}?offset=${offset}&limit=${state.page.limit}`);
  } catch (err) {
    $('rows').innerHTML = `<div class="empty-state">${err.message}</div>`;
    return;
  }
  state.page = { ...state.page, name, offset: page.offset, total: page.total };
  $('data-select').value = name;

  const summary = state.datasets.find((d) => d.name === name);
  $('data-summary').textContent = summary
    ? `${summary.rows} rows · ${bytes(summary.bytes)} · avg ${summary.avg_response_chars} chars · ${summary.teachers.join(', ') || 'no teacher recorded'}`
    : '';
  $('data-range').textContent = page.total
    ? `${page.offset + 1}–${Math.min(page.offset + state.page.limit, page.total)} of ${page.total}`
    : '0 of 0';
  $('btn-prev').disabled = page.offset <= 0;
  $('btn-next').disabled = page.offset + state.page.limit >= page.total;

  const host = $('rows');
  host.innerHTML = '';
  if (!page.rows.length) {
    host.innerHTML = '<div class="empty-state">This dataset is empty.</div>';
    return;
  }
  page.rows.forEach((row, i) => {
    const el = document.createElement('div');
    el.className = 'row';
    el.innerHTML =
      `<div class="idx">${page.offset + i + 1}</div>` +
      `<div class="q">${escapeHtml(row.prompt)}</div>` +
      `<div class="a">${escapeHtml(row.response)}</div>` +
      `<div><button class="ghost small danger" data-drop="${escapeHtml(row.id)}">Delete</button></div>`;
    host.appendChild(el);
  });
  host.querySelectorAll('[data-drop]').forEach((button) => {
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        await api(`/api/datasets/${encodeURIComponent(name)}/rows/${button.dataset.drop}`, { method: 'DELETE' });
        log('dataset', `deleted row ${button.dataset.drop.slice(0, 8)}`);
        await loadDatasets();
        await loadPage(name, state.page.offset);
      } catch (err) {
        log('error', err.message);
        button.disabled = false;
      }
    });
  });
}

function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* ─── event stream ──────────────────────────────────────────────────────── */

let source = null;

function connect() {
  if (source) source.close();
  source = new EventSource('/api/events?replay=250');
  source.onopen = () => setChip('chip-stream', 'ok', 'Telemetry live');
  source.onerror = () => setChip('chip-stream', 'warn', 'Telemetry reconnecting…');
  source.onmessage = (message) => {
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    handle(event);
  };
}

const NOISY = new Set(['dataset.token', 'train.layer_grads', 'job.state']);

function handle(event) {
  if (!NOISY.has(event.kind)) log(event.kind, describe(event));

  switch (event.kind) {
    case 'job.state': applyJobState(event); break;
    case 'job.stopping': $('chip-job').className = 'chip warn'; break;

    case 'dataset.phase':
      state.build.phase = event.phase;
      $('teacher-note').textContent = event.message || event.phase;
      break;

    case 'dataset.prompt':
      state.build.prompts = event.index;
      state.build.target = event.total;
      updateBuildCounter();
      break;

    case 'dataset.generating':
      state.generating += 1;
      openStream(event.index, event.prompt);
      markGenerating(true);
      break;

    case 'dataset.token':
      appendStream(event.index, event.text);
      break;

    case 'dataset.pair':
      state.generating = Math.max(0, state.generating - 1);
      if (!state.generating) markGenerating(false);
      state.build.pairs = event.index;
      state.build.target = event.total;
      closeStream(event.index, 'done');
      spawnPacket('pair');
      updateBuildCounter();
      break;

    case 'dataset.rejected':
      state.generating = Math.max(0, state.generating - 1);
      if (!state.generating) markGenerating(false);
      state.build.rejected += 1;
      closeStream(event.index, 'rejected');
      updateBuildCounter();
      break;

    case 'dataset.done':
      markGenerating(false);
      state.generating = 0;
      $('teacher-note').textContent = event.cancelled
        ? 'build cancelled'
        : `${event.summary?.rows ?? 0} rows in ${state.teacher}`;
      loadDatasets().then(() => loadPage(event.dataset, 0)).catch(() => {});
      break;

    case 'dataset.error':
      markGenerating(false);
      $('teacher-note').textContent = 'build failed';
      break;

    case 'train.loading':
      state.train.active = true;
      $('student-note').textContent = event.message;
      break;

    case 'train.start': startTrainState(event); break;

    case 'train.step': applyStep(event); break;

    case 'train.layer_grads':
      state.train.norms = event.norms || [];
      state.train.normPeak = Math.max(state.train.normPeak * 0.995, ...state.train.norms, 1e-9);
      kick();
      break;

    case 'train.eval':
      state.train.evals.push({ step: event.step, loss: event.eval_loss });
      $('m-eval').textContent = event.eval_loss.toFixed(4);
      drawSpark();
      break;

    case 'train.done':
      state.train.active = false;
      $('student-note').textContent = event.stopped_early
        ? `stopped at step ${event.steps_completed} · loss ${event.final_loss}`
        : `done · ${event.steps_completed} steps · loss ${event.first_loss} → ${event.final_loss} · peak ${bytes(event.peak_vram)}`;
      $('graph-label').textContent = `layer graph — run finished (${event.steps_completed} steps)`;
      loadRuns().catch(() => {});
      break;

    case 'train.error':
      state.train.active = false;
      $('student-note').textContent = `failed: ${event.error}`;
      break;

    case 'train.retry':
    case 'train.warning':
      $('student-note').textContent = event.message;
      break;

    case 'export.done':
      loadExports().catch(() => {});
      switchTab('export');
      break;

    case 'student.download':
      if (event.status === 'done') loadSystem().catch(() => {});
      break;

    default: break;
  }
}

function describe(event) {
  if (event.message) return event.message;
  if (event.error) return event.error;
  switch (event.kind) {
    case 'dataset.prompt': return `${event.index}/${event.total} · ${event.prompt}`;
    case 'dataset.generating': return `answering: ${event.prompt}`;
    case 'dataset.pair': return `pair ${event.index}/${event.total} · ${event.row?.response?.length ?? 0} chars · ${event.tokens_per_second} tok/s`;
    case 'dataset.rejected': return `rejected (${event.reason})`;
    case 'dataset.done': return `${event.summary?.rows ?? 0} rows · ${JSON.stringify(event.stats?.rejected ?? {})}`;
    case 'train.start': return `${event.total_steps} steps · ${event.trainable_params.toLocaleString()} trainable (${event.trainable_pct}%) · ${event.amp_dtype} · 4bit=${event.quantized}`;
    case 'train.step': return `step ${event.step}/${event.total_steps} loss ${event.loss} lr ${event.lr.toExponential(2)} gnorm ${event.grad_norm}`;
    case 'train.eval': return `eval loss ${event.eval_loss}`;
    case 'train.checkpoint': return event.path;
    case 'train.done': return `${event.steps_completed} steps · ${event.first_loss} → ${event.final_loss} · peak ${bytes(event.peak_vram)} · ${duration(event.elapsed)}`;
    case 'export.done': return `${event.name} · ${bytes(event.size_bytes)}`;
    case 'export.start': return `${event.name} (merge=${event.merge})`;
    case 'export.progress': return event.step;
    case 'student.download': return `${event.repo_id}: ${event.status}`;
    default: return '';
  }
}

/* ─── log ───────────────────────────────────────────────────────────────── */

const LOG_MAX = 400;

function log(kind, message) {
  if (!message) return;
  const host = $('log');
  const line = document.createElement('div');
  let level = '';
  if (kind.endsWith('.error') || kind === 'error') level = 'lv-error';
  else if (kind.startsWith('dataset')) level = 'lv-teacher';
  else if (kind.startsWith('train')) level = 'lv-student';
  line.className = level;
  const now = new Date();
  const stamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
  line.innerHTML = `<span class="t">${stamp}</span><span class="k">${kind}</span><span class="m">${escapeHtml(message)}</span>`;
  const atBottom = host.scrollTop + host.clientHeight >= host.scrollHeight - 8;
  host.appendChild(line);
  while (host.childElementCount > LOG_MAX) host.removeChild(host.firstChild);
  if (atBottom) host.scrollTop = host.scrollHeight;
}

/* ─── job state → buttons ───────────────────────────────────────────────── */

function applyJobState(snapshot) {
  state.job = snapshot;
  const current = snapshot && snapshot.current;
  const building = current && current.kind === 'dataset';
  const training = current && current.kind === 'train';
  const exporting = current && current.kind === 'export';

  $('btn-build').disabled = !!current;
  $('btn-train').disabled = !!current;
  $('btn-export').disabled = !!current;
  $('btn-stop-build').disabled = !building;
  $('btn-stop-train').disabled = !training;

  if (current) {
    setChip('chip-job', 'busy', `${current.label} · ${duration(current.elapsed)}`);
  } else if (snapshot && snapshot.last) {
    const last = snapshot.last;
    setChip('chip-job', last.status === 'error' ? 'bad' : '', `${last.label} — ${last.status}`);
  } else {
    setChip('chip-job', '', 'Idle');
  }
  if (!training) state.train.active = false;
  if (exporting) $('export-name').disabled = true; else $('export-name').disabled = false;
}

/* ─── teacher streams ───────────────────────────────────────────────────── */

const STREAM_MAX = 4;
const STREAM_CHARS = 1400;

function openStream(index, prompt) {
  state.streams.set(index, { prompt, text: '', live: true });
  while (state.streams.size > STREAM_MAX) {
    const oldest = [...state.streams.entries()].find(([, s]) => !s.live) || [...state.streams.entries()][0];
    state.streams.delete(oldest[0]);
  }
  renderStreams();
}

function appendStream(index, text) {
  const stream = state.streams.get(index);
  if (!stream) return;
  stream.text = (stream.text + text).slice(-STREAM_CHARS);
  const node = document.querySelector(`.stream[data-index="${index}"] .a`);
  if (node) {
    node.textContent = stream.text;
  } else {
    renderStreams();
  }
}

function closeStream(index, status) {
  const stream = state.streams.get(index);
  if (!stream) return;
  stream.live = false;
  stream.status = status;
  const node = document.querySelector(`.stream[data-index="${index}"]`);
  if (node) node.classList.remove('live');
}

function renderStreams() {
  const host = $('streams');
  host.innerHTML = '';
  if (!state.streams.size) {
    host.innerHTML = '<div class="empty">Nothing generating. Build a dataset to see the teacher answer.</div>';
    $('stream-label').textContent = '';
    return;
  }
  $('stream-label').textContent = `${state.streams.size} slot${state.streams.size > 1 ? 's' : ''}`;
  for (const [index, stream] of state.streams) {
    const el = document.createElement('div');
    el.className = 'stream' + (stream.live ? ' live' : '');
    el.dataset.index = index;
    el.innerHTML = `<div class="q">${escapeHtml(stream.prompt)}</div><div class="a"></div>`;
    el.querySelector('.a').textContent = stream.text || '…';
    host.appendChild(el);
  }
  host.scrollTop = host.scrollHeight;
}

function markGenerating(on) {
  document.querySelectorAll('.model').forEach((row) => {
    row.classList.toggle('generating', on && row.dataset.name === state.teacher);
  });
}

function updateBuildCounter() {
  const b = state.build;
  $('build-counter').textContent = b.target
    ? `${b.pairs}/${b.target} pairs · ${b.prompts} prompts · ${b.rejected} rejected`
    : '';
}

/* ─── training state ────────────────────────────────────────────────────── */

function startTrainState(event) {
  state.train = {
    active: true,
    step: 0,
    total: event.total_steps,
    layers: event.num_layers || 0,
    loraLayers: event.lora_layers || [],
    losses: [], emas: [], evals: [],
    norms: new Array(event.num_layers || 0).fill(0),
    normPeak: 1e-9,
    last: event,
  };
  state.packets = [];
  state.sweeps = [];
  $('student-note').textContent =
    `${event.student} · ${event.num_layers} layers · ${event.trainable_params.toLocaleString()} trainable (${event.trainable_pct}%)`;
  $('graph-label').textContent = `layer graph — ${event.num_layers} layers, ${event.lora_layers.length} with LoRA`;
  ['m-step', 'm-loss', 'm-eval', 'm-lr', 'm-gnorm', 'm-tps', 'm-vram', 'm-eta'].forEach((id) => { $(id).textContent = '—'; });
  const est = event.estimate;
  if (est) $('estimate').textContent = `estimated peak ${bytes(est.total)} (logits ${bytes(est.logits_and_loss)})`;
  drawSpark();
  kick();
}

function applyStep(event) {
  const t = state.train;
  t.active = true;
  t.step = event.step;
  t.total = event.total_steps;
  t.losses.push({ step: event.step, loss: event.loss });
  t.emas.push({ step: event.step, loss: event.loss_ema });
  if (t.losses.length > 4000) { t.losses.shift(); t.emas.shift(); }

  $('m-step').textContent = `${event.step} / ${event.total_steps}`;
  $('m-loss').textContent = event.loss.toFixed(4);
  $('m-lr').textContent = event.lr.toExponential(2);
  $('m-gnorm').textContent = event.grad_norm.toFixed(3);
  $('m-tps').textContent = `${event.tokens_per_second}`;
  $('m-vram').textContent = event.vram_peak ? bytes(event.vram_peak) : 'cpu';
  $('m-eta').textContent = duration(event.eta_seconds);
  $('student-note').textContent = `step ${event.step}/${event.total_steps} · epoch ${event.epoch.toFixed(2)} · ${event.sec_per_step}s/step`;

  spawnPacket('batch', event.label_tokens);
  spawnSweep();
  drawSpark();
}

/* ─── channel packets ───────────────────────────────────────────────────── */

const PACKET_LANES = 7;
const PACKET_MS = 1150;

function spawnPacket(kind, weight) {
  state.packetTotal += 1;
  if (REDUCED_MOTION) { updateChannelMeta(); return; }
  state.packets.push({
    born: performance.now(),
    lane: state.packetTotal % PACKET_LANES,
    kind,
    weight: weight || 0,
  });
  if (state.packets.length > 220) state.packets.splice(0, state.packets.length - 220);
  updateChannelMeta();
  kick();
}

function spawnSweep() {
  if (REDUCED_MOTION) return;
  state.sweeps.push({ born: performance.now() });
  if (state.sweeps.length > 12) state.sweeps.shift();
}

function updateChannelMeta() {
  const b = state.build;
  const t = state.train;
  let label = 'channel idle';
  if (state.generating > 0) label = 'teacher → dataset';
  else if (t.active) label = 'dataset → student';
  $('channel-label').textContent = label;
  $('channel-count').innerHTML = `${b.pairs} pairs<br>${t.step} steps`;
}

/* ─── canvases ──────────────────────────────────────────────────────────── */

function fit(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  return { ctx, w: rect.width, h: rect.height };
}

function drawChannel(now) {
  const canvas = $('channel-canvas');
  const { ctx, w, h } = fit(canvas);
  if (w < 4 || h < 4) return false;

  const top = 16;
  const bottom = h - 16;
  const laneY = (i) => top + ((i + 0.5) * (bottom - top)) / PACKET_LANES;

  // Lane guides: the channel is visible even when empty, but static.
  ctx.strokeStyle = 'rgba(60,66,86,0.55)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 5]);
  for (let i = 0; i < PACKET_LANES; i++) {
    const y = Math.round(laneY(i)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(2, y);
    ctx.lineTo(w - 2, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Anchors at each end, tinted to the zone they belong to.
  ctx.fillStyle = rgba(COLOR.teacher, 0.5);
  ctx.fillRect(0, top - 8, 2, bottom - top + 16);
  ctx.fillStyle = rgba(COLOR.student, 0.5);
  ctx.fillRect(w - 2, top - 8, 2, bottom - top + 16);

  let alive = false;
  const survivors = [];
  for (const packet of state.packets) {
    const p = (now - packet.born) / PACKET_MS;
    if (p >= 1) continue;
    survivors.push(packet);
    alive = true;

    const y = laneY(packet.lane);
    const x = 3 + p * (w - 6);
    const fade = p < 0.12 ? p / 0.12 : p > 0.86 ? (1 - p) / 0.14 : 1;
    const size = packet.kind === 'pair' ? 3.1 : 2.2;
    const color = packet.kind === 'pair'
      ? mix(COLOR.teacher, COLOR.student, p)
      : mix(COLOR.student, COLOR.student, 1);

    // trail
    const tail = Math.max(0, x - (packet.kind === 'pair' ? 26 : 16));
    const gradient = ctx.createLinearGradient(tail, y, x, y);
    gradient.addColorStop(0, 'rgba(0,0,0,0)');
    gradient.addColorStop(1, color);
    ctx.globalAlpha = 0.42 * fade;
    ctx.strokeStyle = gradient;
    ctx.lineWidth = size * 0.8;
    ctx.beginPath();
    ctx.moveTo(tail, y);
    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.globalAlpha = fade;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  state.packets = survivors;
  return alive;
}

const SWEEP_MS = 900;
const GRAPH_NODES = 6;

function drawStudent(now) {
  const canvas = $('student-canvas');
  const { ctx, w, h } = fit(canvas);
  if (w < 4 || h < 4) return false;

  const t = state.train;
  const layers = t.layers || 0;
  if (!layers) {
    ctx.fillStyle = COLOR.faint;
    ctx.font = '11.5px ui-monospace, Consolas, monospace';
    ctx.fillText('layer graph appears once a model geometry is known', 10, 20);
    return false;
  }

  const padX = 26;
  const padY = 10;
  const rows = layers;
  const rowH = (h - padY * 2) / Math.max(1, rows - 1);
  const nodeX = (j) => padX + (j * (w - padX - 12)) / (GRAPH_NODES - 1);
  const rowY = (i) => padY + i * rowH;

  const peak = Math.max(t.normPeak, 1e-9);
  const intensity = (i) => {
    const norm = t.norms[i] || 0;
    return clamp(Math.sqrt(norm / peak), 0, 1);
  };

  // Sweeps: one per real optimiser step, travelling down the stack the way the
  // backward pass does.
  let alive = false;
  const sweeps = [];
  for (const sweep of state.sweeps) {
    const p = (now - sweep.born) / SWEEP_MS;
    if (p >= 1) continue;
    sweeps.push({ p, y: padY + (1 - p) * (h - padY * 2) });
    alive = true;
  }
  state.sweeps = state.sweeps.filter((s) => (now - s.born) / SWEEP_MS < 1);

  const sweepBoost = (y) => {
    let boost = 0;
    for (const sweep of sweeps) {
      const distance = Math.abs(y - sweep.y);
      if (distance < 26) boost = Math.max(boost, (1 - distance / 26) * (1 - sweeps.length * 0.02));
    }
    return clamp(boost, 0, 1);
  };

  // edges
  ctx.lineWidth = 1;
  for (let i = 0; i < rows - 1; i++) {
    const y0 = rowY(i);
    const y1 = rowY(i + 1);
    const a = Math.max(intensity(i), intensity(i + 1));
    const boost = sweepBoost((y0 + y1) / 2);
    const alpha = 0.055 + a * 0.5 + boost * 0.3;
    ctx.strokeStyle = rgba(COLOR.student, clamp(alpha, 0, 0.95));
    ctx.beginPath();
    for (let j = 0; j < GRAPH_NODES; j++) {
      ctx.moveTo(nodeX(j), y0);
      ctx.lineTo(nodeX(j), y1);
      if (j < GRAPH_NODES - 1) {
        ctx.moveTo(nodeX(j), y0);
        ctx.lineTo(nodeX(j + 1), y1);
      }
      if (j > 0) {
        ctx.moveTo(nodeX(j), y0);
        ctx.lineTo(nodeX(j - 1), y1);
      }
    }
    ctx.stroke();
  }

  // nodes + per-layer labels
  ctx.font = '9px ui-monospace, Consolas, monospace';
  ctx.textBaseline = 'middle';
  const labelEvery = rows > 24 ? 4 : rows > 12 ? 2 : 1;
  for (let i = 0; i < rows; i++) {
    const y = rowY(i);
    const a = intensity(i);
    const boost = sweepBoost(y);
    const alpha = clamp(0.14 + a * 0.86 + boost * 0.35, 0, 1);
    const radius = 1.5 + a * 1.9 + boost * 0.8;
    ctx.fillStyle = rgba(COLOR.student, alpha);
    for (let j = 0; j < GRAPH_NODES; j++) {
      ctx.beginPath();
      ctx.arc(nodeX(j), y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    if (i % labelEvery === 0) {
      ctx.fillStyle = rgba(COLOR.student, clamp(0.22 + a * 0.6, 0, 0.9));
      ctx.fillText(String(i), 5, y);
    }
  }

  // The strongest layer is worth naming — it says where learning is concentrated.
  if (t.norms.length) {
    let best = 0;
    for (let i = 1; i < t.norms.length; i++) if (t.norms[i] > t.norms[best]) best = i;
    ctx.fillStyle = COLOR.faint;
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`peak ‖g‖ layer ${best}: ${t.norms[best].toFixed(4)}`, w - 6, h - 8);
    ctx.textAlign = 'left';
  }

  return alive;
}

function drawSpark() {
  const canvas = $('spark-canvas');
  const { ctx, w, h } = fit(canvas);
  const t = state.train;
  const points = t.losses;

  if (!points.length) {
    $('spark-label').textContent = 'loss — no data yet';
    ctx.fillStyle = COLOR.faint;
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.fillText('the loss curve is drawn from real optimiser steps', 8, h / 2);
    return;
  }

  const padL = 38;
  const padR = 6;
  const padT = 8;
  const padB = 14;
  const values = points.map((p) => p.loss).concat(t.evals.map((e) => e.loss));
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-6) { hi += 0.05; lo -= 0.05; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;

  const total = Math.max(t.total, points[points.length - 1].step, 1);
  const px = (step) => padL + ((step - 1) / Math.max(1, total - 1)) * (w - padL - padR);
  const py = (loss) => padT + (1 - (loss - lo) / (hi - lo)) * (h - padT - padB);

  // grid + axis labels
  ctx.strokeStyle = 'rgba(60,66,86,0.5)';
  ctx.fillStyle = COLOR.faint;
  ctx.font = '9px ui-monospace, Consolas, monospace';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (let k = 0; k <= 2; k++) {
    const value = hi - ((hi - lo) * k) / 2;
    const y = Math.round(py(value)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), 4, y);
  }

  // raw loss
  ctx.strokeStyle = rgba(COLOR.student, 0.42);
  ctx.lineWidth = 1;
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(px(p.step), py(p.loss)) : ctx.moveTo(px(p.step), py(p.loss))));
  ctx.stroke();

  // smoothed loss — the trend the raw signal is too noisy to show
  ctx.strokeStyle = rgba(COLOR.student, 0.95);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  t.emas.forEach((p, i) => (i ? ctx.lineTo(px(p.step), py(p.loss)) : ctx.moveTo(px(p.step), py(p.loss))));
  ctx.stroke();

  // eval points
  ctx.fillStyle = COLOR.text;
  for (const point of t.evals) {
    const x = px(point.step);
    const y = py(point.loss);
    ctx.beginPath();
    ctx.moveTo(x, y - 3);
    ctx.lineTo(x + 3, y);
    ctx.lineTo(x, y + 3);
    ctx.lineTo(x - 3, y);
    ctx.closePath();
    ctx.fill();
  }

  const first = points[0].loss;
  const last = points[points.length - 1].loss;
  const ema = t.emas[t.emas.length - 1].loss;
  const evalText = t.evals.length ? ` · eval ${t.evals[t.evals.length - 1].loss.toFixed(4)}` : '';
  $('spark-label').textContent =
    `loss ${first.toFixed(4)} → ${last.toFixed(4)} · ema ${ema.toFixed(4)}${evalText} · ${points.length} steps`;
}

let rafId = null;

function frame(now) {
  rafId = null;
  const a = drawChannel(now);
  const b = drawStudent(now);
  if (a || b) rafId = requestAnimationFrame(frame);
}

function kick() {
  if (rafId === null) rafId = requestAnimationFrame(frame);
}

/* ─── actions ───────────────────────────────────────────────────────────── */

async function buildDataset() {
  const seeds = $('seeds').value.split('\n').map((s) => s.trim()).filter(Boolean);
  const body = {
    name: $('ds-name').value.trim() || 'distilled',
    teacher: state.teacher,
    skill: $('skill').value.trim(),
    seeds,
    target_pairs: Number($('ds-target').value),
    answer_temperature: Number($('ds-temp').value),
    concurrency: Number($('ds-conc').value),
  };
  if (!body.teacher) { log('error', 'Select a teacher model first.'); return; }
  state.build = { active: true, phase: 'start', prompts: 0, pairs: 0, target: body.target_pairs, rejected: 0 };
  state.streams.clear();
  renderStreams();
  try {
    await api('/api/dataset/build', { method: 'POST', body: JSON.stringify(body) });
  } catch (err) {
    log('error', err.message);
  }
}

async function startTraining() {
  const body = {
    dataset: $('tr-dataset').value,
    student: $('student-select').value,
    epochs: Number($('tr-epochs').value),
    max_steps: Number($('tr-steps').value),
    max_seq_len: Number($('tr-seq').value),
    lora_r: Number($('tr-rank').value),
    lora_alpha: Number($('tr-alpha').value),
    lr: Number($('tr-lr').value),
    grad_accum: Number($('tr-accum').value),
    load_in_4bit: $('tr-4bit').checked,
    gradient_checkpointing: $('tr-ckpt').checked,
  };
  if (!body.dataset) { log('error', 'Build a dataset first.'); return; }
  try {
    await api('/api/train/start', { method: 'POST', body: JSON.stringify(body) });
  } catch (err) {
    log('error', err.message);
  }
}

async function stopJob() {
  try {
    await api('/api/train/stop', { method: 'POST' });
  } catch (err) {
    log('error', err.message);
  }
}

async function doExport() {
  const runId = $('export-run').value;
  if (!runId) { log('error', 'No run with an adapter to export.'); return; }
  try {
    await api('/api/export', {
      method: 'POST',
      body: JSON.stringify({
        run_id: runId,
        name: $('export-name').value.trim(),
        merge: $('export-merge').checked,
      }),
    });
  } catch (err) {
    log('error', err.message);
  }
}

let estimateTimer = null;

function onStudentChange() {
  const option = $('student-select').selectedOptions[0];
  if (!option) return;
  const cached = option.dataset.cached === '1';
  $('student-hint').textContent = cached
    ? option.dataset.note
    : `${option.dataset.note} — not downloaded yet; training will fetch it.`;
  scheduleEstimate();
}

function scheduleEstimate() {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(refreshEstimate, 250);
}

async function refreshEstimate() {
  const body = {
    student: $('student-select').value,
    max_seq_len: Number($('tr-seq').value),
    lora_r: Number($('tr-rank').value),
    load_in_4bit: $('tr-4bit').checked,
    gradient_checkpointing: $('tr-ckpt').checked,
  };
  try {
    const result = await api('/api/estimate', { method: 'POST', body: JSON.stringify(body) });
    const est = result.estimate;
    const total = result.vram_total_bytes;
    $('estimate').textContent =
      `estimated peak ${bytes(est.total)}${total ? ` of ${bytes(total)}` : ''}` +
      ` · weights ${bytes(est.weights)} · logits ${bytes(est.logits_and_loss)}` +
      (result.fits ? '' : ' — will not fit');
    // Draw the real layer graph before training starts, using the model's own geometry.
    if (!state.train.active) {
      const layers = result.geometry.num_hidden_layers || 0;
      state.train.layers = layers;
      state.train.norms = new Array(layers).fill(0);
      $('graph-label').textContent = `layer graph — ${layers} layers, idle`;
      kick();
    }
  } catch (err) {
    $('estimate').textContent = err.message.includes('not downloaded')
      ? 'download the base model to see a VRAM estimate'
      : err.message;
  }
}

/* ─── tabs ──────────────────────────────────────────────────────────────── */

function switchTab(name) {
  document.querySelectorAll('.tab[data-tab]').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === name));
  $('drawer').classList.remove('collapsed');
}

/* ─── wiring ────────────────────────────────────────────────────────────── */

function wire() {
  $('btn-build').addEventListener('click', buildDataset);
  $('btn-stop-build').addEventListener('click', stopJob);
  $('btn-train').addEventListener('click', startTraining);
  $('btn-stop-train').addEventListener('click', stopJob);
  $('btn-export').addEventListener('click', doExport);

  $('student-select').addEventListener('change', onStudentChange);
  ['tr-seq', 'tr-rank', 'tr-4bit', 'tr-ckpt'].forEach((id) => $(id).addEventListener('change', scheduleEstimate));

  $('data-select').addEventListener('change', (e) => loadPage(e.target.value, 0));
  $('btn-prev').addEventListener('click', () => loadPage(state.page.name, Math.max(0, state.page.offset - state.page.limit)));
  $('btn-next').addEventListener('click', () => loadPage(state.page.name, state.page.offset + state.page.limit));
  $('btn-drop-dataset').addEventListener('click', async () => {
    const name = $('data-select').value;
    if (!name) return;
    await api(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' });
    log('dataset', `deleted dataset ${name}`);
    state.page.name = '';
    await loadDatasets();
  });

  document.querySelectorAll('.tab[data-tab]').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
  $('btn-collapse').addEventListener('click', () => {
    const drawer = $('drawer');
    drawer.classList.toggle('collapsed');
    $('btn-collapse').textContent = drawer.classList.contains('collapsed') ? '▴' : '▾';
    kick();
  });

  const observer = new ResizeObserver(() => { kick(); drawSpark(); });
  ['channel-canvas', 'student-canvas', 'spark-canvas'].forEach((id) => observer.observe($(id)));
}

async function main() {
  wire();
  renderStreams();
  updateChannelMeta();
  setChip('chip-stream', 'warn', 'Telemetry connecting…');
  connect();
  await loadSystem();
  await loadTeachers();
  await loadRuns();
  await loadDatasets();
  await loadExports();
  drawSpark();
  kick();
  // The teacher list is the only thing that changes without an event of its own
  // (models get pulled or evicted in a terminal), so it is polled, gently.
  setInterval(() => { loadTeachers().catch(() => {}); }, 20000);
}

main().catch((err) => log('error', err.message));
