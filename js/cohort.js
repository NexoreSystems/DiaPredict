import { fetchCohortForecast } from './api.js';

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const GROUPS = ['Низький', 'Помірний', 'Високий'];
const COLORS = { 'Низький': '#198754', 'Помірний': '#e0a800', 'Високий': '#dc3545' };
const BADGE = { 'Низький': 'bg-success', 'Помірний': 'bg-warning text-dark', 'Високий': 'bg-danger' };
const PAGE = 20;
const MAX_POINTS_PER_GROUP = 3000;

let isOpen = false;
let mode = 'risk';
let data = null;
let shown = PAGE;
let levelsChart = null;
let scatterChart = null;
let onOpenPatient = () => {};
let requestId = 0;

export const isCohortOpen = () => isOpen;

const btnHtml = (open) => open ?
    '<svg class="ic me-1"><use href="#i-x"/></svg> Закрити прогноз для всіх' :
    '<svg class="ic me-1"><use href="#i-users"/></svg> Прогноз для всіх пацієнтів';

function setStatus(text, isError = false) {
    const el = $('cohortStatus');
    if (!el) return;
    el.textContent = text;
    el.className = 'small fw-semibold mt-2 ' + (isError ? 'text-danger' : 'text-muted');
}

function on(id, evt, fn) {
    const el = $(id);
    if (el) el.addEventListener(evt, fn);
}

export function initCohort({ onOpen }) {
    onOpenPatient = onOpen;
    on('cohortRefresh', 'click', () => load(true));
    on('cohortCsv', 'click', exportCsv);
    on('cohortFilter', 'input', () => { shown = PAGE;
        renderTable(); });
    on('cohortMore', 'click', () => { shown += PAGE;
        renderTable(); });
    on('cohortBody', 'click', (e) => {
        const row = e.target.closest('tr[data-id]');
        if (row) onOpenPatient(row.dataset.id);
    });
}

export async function openCohort(m) {
    mode = m;
    isOpen = true;
    $('cohortPanel').style.display = 'block';
    $('cohortBtn').innerHTML = btnHtml(true);
    $('cohortTitle').textContent = mode === 'risk' ?
        'Прогноз для всіх пацієнтів скринінгу (без ЦД)' :
        'Прогноз для всіх пацієнтів з ЦД 2 типу';
    $('thCohortHba1c').innerHTML = 'HbA1c зараз' + `<span class="norm">${mode === 'risk' ? 'норма &lt; 5.7%' : 'ціль &lt; 7%'}</span>`;
    $('thCohortRisk').innerHTML = mode === 'risk' ?
        'Ризик діабету, %' :
        'Ризик ускладнень, % <span class="norm">зараз → прогноз</span>';
    $('normCohortMean').textContent = mode === 'risk' ? 'Норма HbA1c < 5.7%' : 'Ціль HbA1c при ЦД2 < 7%';
    $('cohortFilter').value = '';
    shown = PAGE;
    await load(false);
}

export function closeCohort() {
    isOpen = false;
    requestId++; // відповідь, що ще летить, буде проігноровано
    const panel = $('cohortPanel');
    if (panel) panel.style.display = 'none';
    const btn = $('cohortBtn');
    if (btn) btn.innerHTML = btnHtml(false);
}

async function load(refresh) {
    const id = ++requestId;
    setStatus('Розрахунок прогнозу для всіх пацієнтів…');
    try {
        const result = await fetchCohortForecast(mode, refresh);
        if (id !== requestId || !isOpen) return;
        data = result;
        setStatus(data.total ? '' : 'У базі немає пацієнтів для цієї вкладки.');
        render();
    } catch (err) {
        if (id !== requestId) return;
        setStatus(err.message, true);
    }
}

function render() {
    const s = data.summary;
    const pct = (n) => data.total ? Math.round(100 * n / data.total) : 0;
    $('cohortTotal').textContent = data.total;
    $('cohortMean').textContent = data.total ? `${s.mean_hba1c}% → ${s.mean_forecast_hba1c}%` : '—';
    $('cohortWorse').textContent = `${s.worsening} (${pct(s.worsening)}%)`;
    $('cohortHigh').textContent = `${s.levels['Високий']} (${pct(s.levels['Високий'])}%)`;
    renderCharts();
    renderTable();
}

function renderCharts() {
    if (levelsChart) levelsChart.destroy();
    if (scatterChart) scatterChart.destroy();

    levelsChart = new Chart($('cohortLevels'), {
        type: 'doughnut',
        data: {
            labels: GROUPS,
            datasets: [{ data: GROUPS.map(g => data.summary.levels[g]), backgroundColor: GROUPS.map(g => COLORS[g]) }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
    });

    // Не використовуємо Math.min(...масив): на десятках тисяч значень це дає помилку переповнення стека
    let mn = Infinity,
        mx = -Infinity;
    for (const p of data.patients) {
        mn = Math.min(mn, p.hba1c, p.forecast_hba1c);
        mx = Math.max(mx, p.hba1c, p.forecast_hba1c);
    }
    const lo = data.patients.length ? Math.floor(mn * 2) / 2 - 0.5 : 4;
    const hi = data.patients.length ? Math.ceil(mx * 2) / 2 + 0.5 : 12;

    // На графіку не більше MAX_POINTS_PER_GROUP точок на групу (рівномірна вибірка): 30 тис. точок гальмують браузер.
    // Лічильники зверху, таблиця та CSV містять усіх пацієнтів.
    const sample = (arr) => {
        if (arr.length <= MAX_POINTS_PER_GROUP) return arr;
        const step = arr.length / MAX_POINTS_PER_GROUP;
        return Array.from({ length: MAX_POINTS_PER_GROUP }, (_, i) => arr[Math.floor(i * step)]);
    };
    const datasets = GROUPS.map(g => ({
        label: `${g} ризик`,
        data: sample(data.patients.filter(p => p.group === g)).map(p => ({ x: p.hba1c, y: p.forecast_hba1c, id: p.patient_id })),
        backgroundColor: COLORS[g],
        pointRadius: 2,
        pointHoverRadius: 6
    }));
    datasets.push({
        type: 'line',
        label: 'Без змін (y = x)',
        data: [{ x: lo, y: lo }, { x: hi, y: hi }],
        borderColor: '#6b7896',
        borderDash: [5, 5],
        borderWidth: 1.5,
        pointRadius: 0,
        isRef: true
    });

    scatterChart = new Chart($('cohortScatter'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { min: lo, max: hi, title: { display: true, text: 'HbA1c зараз (%)' } },
                y: { min: lo, max: hi, title: { display: true, text: 'Прогноз HbA1c (%)' } }
            },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } },
                tooltip: {
                    filter: item => !item.dataset.isRef,
                    callbacks: { label: item => `${item.raw.id}: ${item.raw.x}% → ${item.raw.y}%` }
                }
            },
            onHover: (e, els, chart) => { chart.canvas.style.cursor = els.length ? 'pointer' : 'default'; },
            onClick: (e, els, chart) => {
                const el = els.find(x => !chart.data.datasets[x.datasetIndex].isRef);
                if (el) onOpenPatient(chart.data.datasets[el.datasetIndex].data[el.index].id);
            }
        }
    });
}

function filteredRows() {
    const q = (($('cohortFilter') || {}).value || '').trim().toLowerCase();
    return q ? data.patients.filter(p => p.patient_id.toLowerCase().includes(q)) : data.patients;
}

function renderTable() {
    if (!data) return;
    const rows = filteredRows();
    $('cohortBody').innerHTML = rows.slice(0, shown).map(p => {
        const dCls = p.delta > 0.2 ? 'text-danger fw-bold' : p.delta < -0.2 ? 'text-success fw-bold' : '';
        const risk = p.risk_forecast != null ? `${p.risk}% → ${p.risk_forecast}%` : `${p.risk}%`;
        return `<tr class="cohort-row" data-id="${escapeHtml(p.patient_id)}">
            <td class="fw-bold text-blue">${escapeHtml(p.patient_id)}</td>
            <td>${p.age} р, ${escapeHtml(p.sex)}</td>
            <td>${p.hba1c}%</td>
            <td class="text-primary fw-bold">${p.forecast_hba1c}%</td>
            <td class="${dCls}">${p.delta > 0 ? '+' : ''}${p.delta}</td>
            <td>${risk}</td>
            <td><span class="badge ${BADGE[p.group]}">${escapeHtml(p.group)}</span></td>
        </tr>`;
    }).join('') || '<tr><td colspan="7" class="text-center text-muted">Нічого не знайдено</td></tr>';

    const more = $('cohortMore');
    const left = rows.length - shown;
    more.style.display = left > 0 ? 'inline-block' : 'none';
    more.textContent = `Показати ще (залишилось ${left})`;
}

function exportCsv() {
    if (!data || !data.patients.length) return;
    const head = ['patient_id', 'age', 'sex', 'visits', 'hba1c', 'forecast_hba1c', 'delta', 'risk_percent', 'risk_forecast_percent', 'level'];
    const lines = data.patients.map(p => [p.patient_id, p.age, p.sex, p.visits, p.hba1c, p.forecast_hba1c,
        p.delta, p.risk, (p.risk_forecast == null ? '' : p.risk_forecast), p.level
    ].join(';'));
    const blob = new Blob(['\uFEFF' + [head.join(';'), ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `forecast_${mode}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}