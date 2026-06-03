/* ===== detection.js — WebSocket progress + result rendering + Kaggle ===== */

const VEHICLE_COLORS = {
  car:        '#00c853',
  truck:      '#f44336',
  bus:        '#2196f3',
  motorcycle: '#ff9800',
  bicycle:    '#ffeb3b',
};

// ─── Progress UI ────────────────────────────────────────────────────────────

function startDetectionUI(jobId) {
  const panel = document.getElementById('progressPanel');
  panel.classList.remove('hidden');
  document.getElementById('resultPanel').classList.add('hidden');

  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/api/v1/videos/ws/${jobId}`);

  ws.onmessage = ({ data }) => {
    const d = JSON.parse(data);
    updateProgress(d);

    if (d.status === 'completed' || d.status === 'failed') {
      ws.close();
      if (d.status === 'completed') {
        loadResult(jobId);
      } else {
        showToast(`Processing failed: ${d.message}`, 'error');
        document.getElementById('progressPanel').classList.add('hidden');
        document.getElementById('uploadZone').classList.remove('hidden');
      }
    }
  };

  ws.onerror = () => {
    // Fallback to polling if WS fails
    pollProgress(jobId);
  };
}

function updateProgress(d) {
  const pct = d.progress || 0;
  document.getElementById('progressFill').style.width = `${pct}%`;
  document.getElementById('progressPct').textContent = `${pct.toFixed(1)}%`;
  document.getElementById('progressLabel').textContent = d.message || 'Processing…';
  document.getElementById('frameCount').textContent = `${d.current_frame || 0} / ${d.total_frames || 0}`;
  document.getElementById('elapsedTime').textContent = `${d.elapsed_sec || 0}s`;
  document.getElementById('etaTime').textContent = d.eta_sec != null ? `${d.eta_sec}s` : '—';

  const statusMap = { queued: 'badge-queued', processing: 'badge-processing', completed: 'badge-completed', failed: 'badge-failed' };
  document.getElementById('jobStatusBadge').innerHTML =
    `<span class="badge ${statusMap[d.status] || ''}">${d.status}</span>`;
}

async function pollProgress(jobId) {
  const interval = setInterval(async () => {
    try {
      const r = await fetch(`/api/v1/videos/${jobId}/status`);
      const d = await r.json();

      const prog = {
        status:        d.status,
        progress:      d.status === 'completed' ? 100 : 50,
        current_frame: d.video_metadata?.total_frames || 0,
        total_frames:  d.video_metadata?.total_frames || 0,
        message:       d.error || d.status,
        elapsed_sec:   0,
        eta_sec:       null,
      };
      updateProgress(prog);

      if (d.status === 'completed') {
        clearInterval(interval);
        loadResult(jobId);
      } else if (d.status === 'failed') {
        clearInterval(interval);
        showToast(`Failed: ${d.error}`, 'error');
      }
    } catch {
      clearInterval(interval);
    }
  }, 2000);
}

// ─── Result rendering ───────────────────────────────────────────────────────

async function loadResult(jobId) {
  try {
    const r = await fetch(`/api/v1/videos/${jobId}/status`);
    const job = await r.json();

    document.getElementById('progressPanel').classList.add('hidden');
    const panel = document.getElementById('resultPanel');
    panel.classList.remove('hidden');

    renderStats(job.summary);
    renderBreakdown(job.summary);
    renderMeta(job.video_metadata, job.summary);

    document.getElementById('downloadBtn').onclick = () => {
      window.open(`/api/v1/videos/${jobId}/download`, '_blank');
    };
  } catch {
    showToast('Could not load results', 'error');
  }
}

function renderStats(s) {
  if (!s) return;
  const cards = [
    { val: s.total_detections, label: 'Total Detections' },
    { val: s.peak_count, label: 'Peak Count' },
    { val: s.avg_detections_per_frame.toFixed(2), label: 'Avg / Frame' },
    { val: s.processing_fps, label: 'Proc. FPS' },
    { val: s.processed_frames, label: 'Frames Processed' },
  ];
  document.getElementById('statsGrid').innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="s-val">${c.val}</div>
      <div class="s-label">${c.label}</div>
    </div>
  `).join('');
}

function renderBreakdown(s) {
  if (!s || !s.vehicle_counts) return;
  const counts = s.vehicle_counts;
  const max = Math.max(...Object.values(counts), 1);

  document.getElementById('breakdownBars').innerHTML = Object.entries(counts)
    .sort(([, a], [, b]) => b - a)
    .map(([label, count]) => {
      const pct = (count / max * 100).toFixed(1);
      const color = VEHICLE_COLORS[label] || '#8b949e';
      return `
        <div class="breakdown-row" style="display:flex;align-items:center;gap:.75rem">
          <span class="breakdown-label">${label}</span>
          <div class="breakdown-track" style="flex:1;background:var(--bg3);border-radius:999px;height:8px">
            <div class="breakdown-fill" style="width:${pct}%;background:${color};height:100%;border-radius:999px"></div>
          </div>
          <span class="breakdown-count">${count}</span>
        </div>`;
    }).join('');
}

function renderMeta(meta, summary) {
  if (!meta) return;
  const items = [
    { k: 'Filename',   v: meta.filename },
    { k: 'Duration',   v: `${meta.duration_sec}s` },
    { k: 'Resolution', v: `${meta.width} × ${meta.height}` },
    { k: 'FPS',        v: meta.fps },
    { k: 'Total Frames', v: meta.total_frames },
    { k: 'File Size',  v: `${(meta.size_bytes / (1024*1024)).toFixed(1)} MB` },
  ];
  document.getElementById('metaGrid').innerHTML = items.map(i => `
    <div class="meta-item">
      <div class="m-key">${i.k}</div>
      <div class="m-val">${i.v}</div>
    </div>
  `).join('');
}

// ─── Kaggle UI ───────────────────────────────────────────────────────────────

document.getElementById('kaggleSearchBtn').addEventListener('click', searchKaggle);
document.getElementById('kaggleQuery').addEventListener('keydown', e => {
  if (e.key === 'Enter') searchKaggle();
});

async function searchKaggle() {
  const q = document.getElementById('kaggleQuery').value.trim();
  if (!q) return;
  const el = document.getElementById('kaggleResults');
  el.innerHTML = '<div class="empty-state">Searching…</div>';

  try {
    const r = await fetch(`/api/v1/datasets/search?q=${encodeURIComponent(q)}&limit=20`);
    if (!r.ok) throw new Error((await r.json()).detail);
    const datasets = await r.json();

    if (!datasets.length) {
      el.innerHTML = '<div class="empty-state">No datasets found.</div>';
      return;
    }
    el.innerHTML = datasets.map(ds => `
      <div class="dataset-card">
        <div class="dataset-header">
          <div>
            <div class="dataset-title">${ds.title}</div>
            <div class="dataset-ref">${ds.ref}</div>
          </div>
          <div class="dataset-actions">
            <a href="${ds.url}" target="_blank" class="btn btn-sm btn-secondary">View</a>
            <button class="btn btn-sm btn-primary" onclick="downloadDataset('${ds.ref}', this)">Download</button>
          </div>
        </div>
        <div class="dataset-meta">
          <span>${ds.size_mb} MB</span>
          <span>${ds.download_count.toLocaleString()} downloads</span>
          <span>${ds.vote_count} votes</span>
          <span>${ds.last_updated}</span>
        </div>
        ${ds.tags.length ? `<div class="dataset-tags">${ds.tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>` : ''}
      </div>
    `).join('');
  } catch (err) {
    el.innerHTML = `<div class="empty-state" style="color:var(--danger)">${err.message}</div>`;
  }
}

async function downloadDataset(ref, btn) {
  btn.disabled = true;
  btn.textContent = 'Downloading…';
  try {
    const r = await fetch(`/api/v1/datasets/download?ref=${encodeURIComponent(ref)}`, { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail);
    showToast('Dataset downloaded successfully!');
    loadLocalDatasets();
  } catch (err) {
    showToast(`Download failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Download';
  }
}

async function loadLocalDatasets() {
  const el = document.getElementById('localDatasets');
  try {
    const r = await fetch('/api/v1/datasets/local');
    const datasets = await r.json();
    if (!datasets.length) {
      el.innerHTML = '<div class="empty-state">No datasets downloaded yet.</div>';
      return;
    }
    el.innerHTML = datasets.map(ds => `
      <div class="dataset-card">
        <div class="dataset-header">
          <div>
            <div class="dataset-title">${ds.name}</div>
            <div class="dataset-ref">${ds.path}</div>
          </div>
        </div>
        <div class="dataset-meta">
          <span>${ds.file_count} files</span>
          <span>${ds.size_mb} MB</span>
        </div>
      </div>
    `).join('');
  } catch {
    el.innerHTML = '<div class="empty-state">Failed to load local datasets.</div>';
  }
}

document.getElementById('refreshLocalBtn').addEventListener('click', loadLocalDatasets);
