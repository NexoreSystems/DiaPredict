import { fetchPatients, fetchPatientHistory, saveLifestyle } from './api.js';
import { handleLabSubmit, initLabModule, handlePatientModeChange, handleExistingPatientSelect } from './lab.js';
import { renderCharts, switchChart } from './chart_utils.js';

let currentMode = 'risk';
let globalPatientData = null;

const $ = (id) => document.getElementById(id);
const setText = (id, text) => { const el = $(id); if (el) el.innerText = text; };
const setDisplay = (id, value) => { const el = $(id); if (el) el.style.display = value; };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(Number(v));
const fmt = (v, digits = 1, suffix = '') => (isNum(v) ? Number(v).toFixed(digits) + suffix : '—');

// ── Пороги для статусів показників: [текст, рівень] ──
const LEVEL_CLASS = { ok: 'text-success', warn: 'text-warning', bad: 'text-danger' };
const bmiStatus = (v) => v < 25 ? ['[В нормі]', 'ok'] : v < 30 ? ['[Надмірна вага]', 'warn'] : ['[Ожиріння]', 'bad'];
const glucoseStatus = (v) => v <= 5.6 ? ['[В нормі]', 'ok'] : v <= 6.9 ? ['[Переддіабет]', 'warn'] : ['[Вище норми]', 'bad'];
const bpStatus = (v) => v < 130 ? ['[В нормі]', 'ok'] : ['[Високий АТ]', 'bad'];

function setStatus(id, [text, level]) {
    const el = $(id);
    if (!el) return;
    el.innerText = text;
    el.className = `status-tag ${LEVEL_CLASS[level]}`;
}

const RISK_BADGE_CLASS = [
    ['Високий', 'badge badge-forecast bg-danger'],
    ['Помірний', 'badge badge-forecast bg-warning text-dark'],
];
const riskBadgeClass = (level) =>
    (RISK_BADGE_CLASS.find(([word]) => level.includes(word)) || [, 'badge badge-forecast bg-success'])[1];

const TAB_IDS = { risk: 'tabRisk', complication: 'tabComplication', lab: 'tabLab' };
const PAGE_TITLES = { risk: 'Скринінг ризику діабету', complication: 'Ризик судинних ускладнень (ЦД 2 типу)', lab: 'Введення лабораторних даних' };

document.addEventListener('DOMContentLoaded', () => {
    setText('printDate', new Date().toLocaleDateString('uk-UA'));
    setText('topDate', new Date().toLocaleDateString('uk-UA', { day: 'numeric', month: 'long', year: 'numeric' }));
    const labForm = $('labForm');
    if (labForm) labForm.addEventListener('submit', handleLabSubmit);

    // Прив'язка до window для доступу з HTML (onclick / onchange)
    Object.assign(window, {
        switchMode,
        switchChart,
        loadPatientData,
        saveLifestyle: saveLifestyleHandler,
        setLabMode: handlePatientModeChange,
        onExistingPatientChange: handleExistingPatientSelect,
        printReport: () => window.print(),
    });

    const search = $('patientSearch');
    if (search) {
        let timer = null;
        search.addEventListener('input', () => {
            clearTimeout(timer);
            timer = setTimeout(() => loadPatientList(search.value.trim()), 250);
        });
    }

    switchMode('risk');
});

async function switchMode(mode) {
    currentMode = mode;
    Object.entries(TAB_IDS).forEach(([key, id]) => {
        const tab = $(id);
        if (tab) tab.classList.toggle('active', mode === key);
    });
    setText('pageTitle', PAGE_TITLES[mode]);

    const isLab = mode === 'lab';
    setDisplay('patientSelectCard', isLab ? 'none' : 'block');
    setDisplay('labModule', isLab ? 'block' : 'none');
    setDisplay('mainDashboard', 'none');

    if (isLab) {
        await initLabModule();
        return;
    }

    const isRisk = mode === 'risk';
    const selectLabel = $('selectLabel');
    if (selectLabel) {
        selectLabel.innerHTML = isRisk ?
            '<svg class="ic me-2"><use href="#i-shield"/></svg> Пацієнт для оцінки ризику (без діагнозу ЦД)' :
            '<svg class="ic me-2"><use href="#i-activity"/></svg> Пацієнт з підтвердженим ЦД 2 типу (оцінка ускладнень)';
    }
    const riskTitle = $('riskTitle');
    if (riskTitle) {
        riskTitle.innerHTML = isRisk ?
            '<svg class="ic me-1"><use href="#i-shield"/></svg> Ризик захворюваності' :
            '<svg class="ic me-1"><use href="#i-activity"/></svg> Ризик судинних ускладнень';
    }
    const patientInfo = $('patientInfoBadge');
    if (patientInfo) patientInfo.innerHTML = '<svg class="ic me-1"><use href="#i-info"/></svg> Оберіть картку для завантаження аналізів';

    const search = $('patientSearch');
    if (search) search.value = '';
    await loadPatientList('');
}

// Завантаження списку пацієнтів (з пошуком за частиною ID на сервері)
let listRequestId = 0;
async function loadPatientList(query) {
    const requestId = ++listRequestId;
    try {
        const patients = await fetchPatients(currentMode, query);
        if (requestId !== listRequestId) return; // застарілу відповідь ігноруємо
        const select = $('patientSelect');
        if (!select) return;

        const head = patients.length ? '-- Оберіть ID пацієнта --' : 'Нічого не знайдено';
        select.innerHTML = `<option value="">${head}</option>` + patients.map(p =>
            `<option value="${escapeHtml(p.patient_id)}">${escapeHtml(p.patient_id)} | ${p.age}р, ${escapeHtml(p.sex)} | Візитів: ${p.visits} (${escapeHtml(p.status)})</option>`
        ).join('');

        const exact = query && patients.find(p => p.patient_id.toLowerCase() === query.toLowerCase());
        if (exact) {
            select.value = exact.patient_id;
            loadPatientData();
        }
    } catch (err) {
        console.error('Помилка завантаження списку пацієнтів:', err);
        const select = $('patientSelect');
        if (select) select.innerHTML = `<option value="">${escapeHtml(err.message)}</option>`;
    }
}

const setSelect = (id, value) => { const el = $(id); if (el) el.value = value === 'Не вказано' ? '' : (value || ''); };

async function loadPatientData() {
    const select = $('patientSelect');
    if (!select) return;

    const pid = select.value;
    if (!pid) {
        setDisplay('mainDashboard', 'none');
        return;
    }

    try {
        const data = await fetchPatientHistory(pid, currentMode);
        globalPatientData = data;
        setDisplay('mainDashboard', 'block');

        const { profile: prof, ml_predictions: ml, lifestyle: life = {}, clinical_summary: summary = {} } = data;

        setText('valAge', prof.age + ' р.');
        setText('valBmi', fmt(prof.bmi, 1, ' кг/м²'));
        setStatus('statusBmi', bmiStatus(prof.bmi));
        setText('valGlucose', fmt(prof.glucose_fasting, 2, ' ммоль/л'));
        setStatus('statusGlucose', glucoseStatus(prof.glucose_fasting));
        setText('valBp', fmt(prof.systolic_bp, 0, ' мм рт.ст.'));
        setStatus('statusBp', bpStatus(prof.systolic_bp));

        // Спосіб життя — реальні дані, які вводить лікар (не генеруються)
        setSelect('lifeSmoking', life.smoking);
        setSelect('lifeAlcohol', life.alcohol);
        setSelect('lifeFamily', life.family_history);
        setText('lifeMsg', '');

        setText('valHba1cCurrent', fmt(prof.HbA1c, 1, '%'));
        setText('dataQuality', (isNum(data.data_quality) ? data.data_quality : 100) + '%');
        $('patientInfoBadge').innerHTML =
            `<span class="badge bg-dark p-2 fs-6">ID: <b>${escapeHtml(data.patient_id)}</b> | Стать: <b>${escapeHtml(prof.sex)}</b> | Статус: <b>${data.is_diabetic ? 'Підтверджений ЦД2' : 'Без діабету'}</b></span>`;

        const comp = ml.complication;
        if (comp) { // Модуль 2: індекс ускладнень замість ймовірності діабету
            setText('riskScore', comp.score + ' / 100');
            setText('riskSubtitle', `Індекс за правилами (не ML) · прогноз: ${comp.forecast_score}`);
        } else {
            setText('riskScore', ml.risk_score + '%');
            setText('riskSubtitle', 'Ймовірність діабету (ML Tier 1)');
        }
        setText('predictedHbA1c', ml.predicted_hba1c + '%');
        setText('recommendationText', '💡 Порада СППР: ' + ml.recommendation);

        const badge = $('riskBadge');
        badge.innerText = ml.risk_level;
        badge.className = riskBadgeClass(ml.risk_level);

        // Текст висновків формується на сервері і містить <b>, тому вставляється як HTML
        const toListHtml = (items = []) => items.map(item => `<li class="mb-1">${item}</li>`).join('');
        $('conclusionsList').innerHTML = toListHtml(summary.conclusions);
        $('actionsList').innerHTML = toListHtml(summary.recommended_actions);

        $('historyTable').querySelector('tbody').innerHTML = data.history.map(row => {
            const hba = isNum(row.HbA1c) ? Number(row.HbA1c) : null;
            const glu = isNum(row.glucose_fasting) ? Number(row.glucose_fasting) : null;
            const abnormal = (hba !== null && hba >= 6.5) || (glu !== null && glu >= 6.1);
            return `
                <tr class="${abnormal ? 'table-danger' : ''}">
                    <td>${escapeHtml(row.formatted_date || row.test_date)}</td>
                    <td class="fw-bold ${hba !== null && hba >= 6.5 ? 'text-danger' : 'text-success'}">${fmt(hba, 1, '%')}</td>
                    <td class="text-primary fw-bold">${fmt(glu, 2)}</td>
                </tr>`;
        }).join('');

        renderCharts(data);
    } catch (err) {
        console.error('Помилка при завантаженні даних пацієнта:', err);
        setDisplay('mainDashboard', 'none');
        const info = $('patientInfoBadge');
        if (info) info.innerHTML = `<span class="text-danger fw-semibold">${escapeHtml(err.message)}</span>`;
    }
}

// Збереження способу життя, обраного лікарем
async function saveLifestyleHandler() {
    if (!globalPatientData) return;
    const btn = $('lifeSaveBtn');
    if (btn) btn.disabled = true;
    const { ok, data } = await saveLifestyle(globalPatientData.patient_id, {
        smoking: $('lifeSmoking').value || null,
        alcohol: $('lifeAlcohol').value || null,
        family_history: $('lifeFamily').value || null,
    });
    if (btn) btn.disabled = false;
    if (ok) {
        await loadPatientData(); // оновлюємо висновки (наприклад, вплив куріння)
        setText('lifeMsg', '✅ Збережено');
    } else {
        setText('lifeMsg', '❌ ' + data.message);
    }
}