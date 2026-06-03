/* ===== upload.js — drag & drop, file selection, upload trigger ===== */

const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');
const browseBtn  = document.getElementById('browseBtn');
const fileInfo   = document.getElementById('fileInfo');
const fileName   = document.getElementById('fileName');
const fileSize   = document.getElementById('fileSize');
const clearBtn   = document.getElementById('clearFileBtn');
const startBtn   = document.getElementById('startBtn');

let selectedFile = null;

function formatBytes(b) {
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function setFile(file) {
  const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast(`Unsupported file type: ${ext}`, 'error');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    showToast('File exceeds 500 MB limit', 'error');
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  uploadZone.classList.add('hidden');
  fileInfo.classList.remove('hidden');
}

// Browse button
browseBtn.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('click', e => { if (e.target !== browseBtn) fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

// Drag & drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

// Clear
clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  fileInfo.classList.add('hidden');
  uploadZone.classList.remove('hidden');
});

// Start detection
startBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  startBtn.disabled = true;
  startBtn.textContent = 'Uploading...';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const resp = await fetch('/api/v1/videos/upload', { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    const { job_id } = await resp.json();
    showToast('Upload successful — processing started!');
    fileInfo.classList.add('hidden');
    startDetectionUI(job_id);
  } catch (err) {
    showToast(err.message, 'error');
    startBtn.disabled = false;
    startBtn.textContent = 'Start Detection';
  }
});

// New job button
document.getElementById('newJobBtn').addEventListener('click', () => {
  document.getElementById('resultPanel').classList.add('hidden');
  document.getElementById('progressPanel').classList.add('hidden');
  uploadZone.classList.remove('hidden');
  selectedFile = null;
  fileInput.value = '';
  startBtn.disabled = false;
  startBtn.textContent = 'Start Detection';
});
