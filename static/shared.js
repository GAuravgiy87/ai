/* =============================================================
   shared.js — theme, nav highlight, toast, clock, SSE
   Loaded on every page: <script src="/static/shared.js"></script>
   ============================================================= */

// ---- Theme ----
function toggleTheme() {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    if (typeof onThemeChange === 'function') onThemeChange(next); // hook for chart pages
}

function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    const cb = document.getElementById('theme-checkbox');
    if (cb) cb.checked = (saved === 'dark');
}

// ---- Active Nav ----
const _NAV_MAP = {
    '/':                      'nav-live',
    '/dashboard':             'nav-dashboard',
    '/recordings_page':       'nav-recordings',
    '/detection_logs':        'nav-logs',
    '/people':                'nav-people',
    '/search':                'nav-search',
    '/journey':               'nav-journey',
    '/analytics':             'nav-analytics',
    '/registered_detections': 'nav-reg-logs',
    '/cameras':               'nav-cameras',
    '/add_camera':            'nav-add-camera',
    '/system_logs':           'nav-syslogs',
};
function setActiveNav() {
    const id = _NAV_MAP[window.location.pathname];
    if (id) document.getElementById(id)?.classList.add('active');
}

// ---- Toast ----
function showToast(data) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <img src="${data.thumbnail}" class="toast-thumb" onerror="this.style.display='none'">
        <div class="toast-content">
            <span class="toast-title">${data.target} spotted</span>
            <span class="toast-meta">${data.camera} &bull; ${data.time}</span>
        </div>
        <div class="toast-progress"></div>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('active'));
    toast.querySelector('.toast-progress').animate(
        [{ width: '100%' }, { width: '0%' }],
        { duration: 5000, fill: 'forwards' }
    );
    setTimeout(() => {
        toast.classList.remove('active');
        setTimeout(() => toast.remove(), 600);
    }, 5000);
}

// ---- SSE Notifications ----
function startNotificationListener() {
    const es = new EventSource('/api/notifications/stream');
    es.onmessage = e => {
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'detection') showToast(data);
        } catch (_) {}
    };
    es.onerror = () => { es.close(); setTimeout(startNotificationListener, 5000); };
}

// ---- Server-synced IST Clock ----
let _serverOffset = 0;
async function syncServerTime() {
    try {
        const t0 = Date.now();
        const res = await fetch('/api/server_time');
        const t1 = Date.now();
        if (res.ok) {
            const { timestamp_ms } = await res.json();
            _serverOffset = timestamp_ms - t0 + Math.round((t1 - t0) / 2);
        }
    } catch (_) {}
}
function getServerIST() { return new Date(Date.now() + _serverOffset); }
function updateLiveClock(elementId = 'live-clock') {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = 'SYNC: ' + getServerIST().toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
    }) + ' IST';
}

// ---- Boot (runs on every page) ----
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setActiveNav();
    startNotificationListener();
    syncServerTime().then(() => {
        updateLiveClock();
        setInterval(updateLiveClock, 1000);
        setInterval(syncServerTime, 5 * 60 * 1000);
    });
});
