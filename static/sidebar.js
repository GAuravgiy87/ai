/**
 * sidebar.js — loads the shared sidebar partial and handles theme + active nav.
 * Include this script on every page AFTER the #sidebar-mount placeholder.
 */

// ── Theme ──────────────────────────────────────────────────────────────────
function toggleTheme() {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    const cb = document.getElementById('theme-checkbox');
    if (cb) cb.checked = (next === 'dark');
}

// ── Active nav highlight ───────────────────────────────────────────────────
function _setActiveNav() {
    const path = window.location.pathname;
    document.querySelectorAll('.sidebar-menu a[data-nav]').forEach(a => {
        a.classList.remove('active');
        if (a.getAttribute('href') === path) a.classList.add('active');
    });
}

// ── Load sidebar HTML into mount point ────────────────────────────────────
async function loadSidebar(mountId) {
    const mount = document.getElementById(mountId || 'sidebar-mount');
    if (!mount) return;
    try {
        const res = await fetch('/sidebar');
        if (!res.ok) return;
        mount.outerHTML = await res.text();
    } catch (e) { return; }

    // Sync theme checkbox
    const cb = document.getElementById('theme-checkbox');
    if (cb) cb.checked = (localStorage.getItem('theme') || 'light') === 'dark';

    _setActiveNav();
}

// Auto-run on DOMContentLoaded if mount point exists
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('sidebar-mount')) loadSidebar('sidebar-mount');
    else _setActiveNav(); // page already has sidebar injected server-side
});
