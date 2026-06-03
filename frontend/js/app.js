/* ===== app.js — tab routing, health check, toast ===== */

const API = '/api/v1';

// --- Tab routing ---
document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

    if (btn.dataset.tab === 'results') loadJobs();
    if (btn.dataset.tab === 'kaggle') loadLocalDatasets();
  });
});

// --- Health check ---
async function checkHealth() {
  const dot  = document.getElementById('apiStatus');
  const text = document.getElementById('apiStatusText');
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    if (d.status === 'ok') {
      dot.className  = 'status-dot ok';
      text.textContent = d.model_ready ? 'API ready' : 'Model missing – run download_models.py';
      if (!d.model_ready) {
        showToast('Model not found. Run: python scripts/download_models.py', 'error');
      }
    } else {
      throw new Error();
    }
  } catch {
    dot.className  = 'status-dot fail';
    text.textContent = 'API offline';
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// --- Toast ---
let _toastTimer;
function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}

// --- Jobs list (results tab) ---
async function loadJobs() {
  const el = document.getElementById('jobsList');
  try {
    const jobs = await (await fetch(`${API}/videos/`)).json();
    if (!jobs.length) {
      el.innerHTML = '<div class="empty-state">No jobs yet. Upload a video to get started.</div>';
      return;
    }
    el.innerHTML = jobs.map(j => `
      <div class="job-item">
        <span class="job-id">${j.job_id}</span>
        <div class="job-meta">
          <span class="badge badge-${j.status}">${j.status}</span>
          ${j.video_metadata ? `<span style="font-size:.85rem;color:var(--text-muted)">${j.video_metadata.filename}</span>` : ''}
          ${j.summary ? `<span style="font-size:.85rem">${j.summary.total_detections} detections</span>` : ''}
          <span style="font-size:.8rem;color:var(--text-muted)">${new Date(j.created_at).toLocaleString()}</span>
        </div>
        <div class="job-actions">
          ${j.status === 'completed' ? `<button class="btn btn-sm btn-primary" onclick="downloadJob('${j.job_id}')">Download</button>` : ''}
        </div>
      </div>
    `).join('');
  } catch {
    el.innerHTML = '<div class="empty-state">Failed to load jobs.</div>';
  }
}

function downloadJob(jobId) {
  window.open(`${API}/videos/${jobId}/download`, '_blank');
}

document.getElementById('refreshJobsBtn').addEventListener('click', loadJobs);
