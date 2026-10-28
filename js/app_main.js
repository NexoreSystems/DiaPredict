import { fetchPatients, fetchPatientHistory, saveLifestyle } from './api.js';
import { handleLabSubmit, initLabModule, handlePatientModeChange, handleExistingPatientSelect } from './lab.js';
import { renderCharts, switchChart } from './chart_utils.js';
import { initCohort, openCohort, closeCohort, isCohortOpen } from './cohort.js';

let currentMode = 'risk';
let globalPatientData = null;

const $ = (id) => document.getElementById(id);
const setText = (id, text) => { const el = $(id); if (el) el.innerText = text; };
const setDisplay = (id, value) => { const el = $(id); if (el) el.style.display = value; };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(Number(v));
const fmt = (v, digits = 1, suffix = '') => (isNum(v) ? Number(v).toFixed(digits) + suffix : '—');
const icon = (id, cls = 'me-1') => `<svg class="ic ${cls}"><use href="#${id}"/></svg>`;
const isDiabetic = () => currentMode === 'complication';

// ── Статуси показників: [текст, рівень]. Для ЦД2 — цільові значення, а не діагностичні пороги ──
const LEVEL_CLASS = { ok: 'text-success', warn: 'text-warning', bad: 'text-danger', na: 'text-muted' };
const NO_DATA = ['[немає даних]', 'na'];
const STATUS = {
    bmi: (v) => v < 18.5 ? ['[Дефіцит маси]', 'warn'] : v < 25 ? ['[В нормі]', 'ok'] : v < 30 ? ['[Надмірна вага]', 'warn'] : ['[Ожиріння]', 'bad'],
    glucose: (v, d) => d ?
        (v < 3.9 ? ['[Гіпоглікемія]', 'bad'] : v < 4.4 ? ['[Нижче цілі]', 'warn'] : v <= 7.2 ? ['[У цілі]', 'ok'] : v <= 10 ? ['[Вище цілі]', 'warn'] : ['[Значно вище цілі]', 'bad']) :
        (v < 5.6 ? ['[В нормі]', 'ok'] : v < 7.0 ? ['[Переддіабет]', 'warn'] : ['[Рівень діабету]', 'bad']),
    hba1c: (v, d) => d ?
        (v < 7 ? ['[У цілі]', 'ok'] : v < 8 ? ['[Вище цілі]', 'warn'] : ['[Декомпенсація]', 'bad']) :
        (v < 5.7 ? ['[В нормі]', 'ok'] : v < 6.5 ? ['[Переддіабет]', 'warn'] : ['[Рівень діабету]', 'bad']),
    bp: (v) => v < 130 ? ['[В нормі]', 'ok'] : v < 140 ? ['[Підвищений]', 'warn'] : ['[Високий АТ]', 'bad'],
    trig: (v) => v < 1.7 ? ['[В нормі]', 'ok'] : v < 2.3 ? ['[Погранично]', 'warn'] : ['[Підвищені]', 'bad'],
};
const statusOf = (fn, v, ...rest) => (isNum(v) ? fn(Number(v), ...rest) : NO_DATA);

function setStatus(id, [text, level]) {
    const el = $(id);
    if (!el) return;
    el.innerText = text;
    el.className = `status-tag ${LEVEL_CLASS[level]}`;
}

// ── Норми, що показуються під показниками (залежать від вкладки) ──
const NORMS = {
    risk: {
        normAge: 'Скринінг діабету з 45 р.',
        normBmi: 'Норма 18.5–24.9',
        normGlucose: 'Норма < 5.6 ммоль/л',
        normBp: 'Норма < 130/80',
        normHba1c: 'Норма < 5.7%',
        normTrig: 'Норма < 1.7 ммоль/л',
        normKpiHba1c: 'Норма < 5.7% · предіабет 5.7–6.4% · ЦД ≥ 6.5%',
        normKpiForecast: 'Норма < 5.7% · ЦД ≥ 6.5%',
        normKpiRisk: 'Шкала: < 30% низький · 30–59% помірний · ≥ 60% високий',
        normsNote: 'Норми МОЗ/ВООЗ: ІМТ 18.5–24.9 · глюкоза натще < 5.6 ммоль/л · HbA1c < 5.7% · АТ < 130/80 · тригліцериди < 1.7 ммоль/л',
        thHba1c: 'HbA1c<span class="norm">норма &lt; 5.7%</span>',
        thGlucose: 'Глюкоза<span class="norm">норма &lt; 5.6</span>',
    },
    complication: {
        normAge: 'Ризик ускладнень зростає з 45 р.',
        normBmi: 'Норма 18.5–24.9',
        normGlucose: 'Ціль 4.4–7.2 ммоль/л',
        normBp: 'Ціль < 130/80',
        normHba1c: 'Ціль < 7.0%',
        normTrig: 'Норма < 1.7 ммоль/л',
        normKpiHba1c: 'Ціль при ЦД 2 типу: < 7.0%',
        normKpiForecast: 'Ціль при ЦД 2 типу: < 7.0%',
        normKpiRisk: 'Шкала: < 25% низький · 25–49% помірний · ≥ 50% високий',
        normsNote: 'Цільові значення при ЦД 2 типу (індивідуалізуються лікарем): HbA1c < 7% · глюкоза натще 4.4–7.2 ммоль/л · АТ < 130/80 · ІМТ 18.5–24.9 · тригліцериди < 1.7 ммоль/л',
        thHba1c: 'HbA1c<span class="norm">ціль &lt; 7%</span>',
        thGlucose: 'Глюкоза<span class="norm">ціль 4.4–7.2</span>',
    },
};

function applyNorms() {
    const set = NORMS[isDiabetic() ? 'complication' : 'risk'];
    Object.entries(set).forEach(([id, text]) => {
        if (id === 'thHba1c' || id === 'thGlucose') {
            const el = $(id);
            if (el) el.innerHTML = text;
        } else {
            setText(id, text);
        }
    });
}

const RISK_BADGE_CLASS = [
    ['Високий', 'badge badge-forecast bg-danger'],
    ['Помірний', 'badge badge-forecast bg-warning text-dark'],
];
const riskBadgeClass = (level) =>
    (RISK_BADGE_CLASS.find(([word]) => level.includes(word)) || [, 'badge badge-forecast bg-success'])[1];

// ── Спосіб життя: іконки та кольорові позначки ──
const LIFE_FIELDS = [{
        select: 'lifeSmoking',
        chip: 'chipSmoking',
        ic: 'i-smoke',
        levels: { 'Не курить': 'ok', 'Колишній курець': 'warn', 'Активний курець': 'bad' }
    },
    {
        select: 'lifeAlcohol',
        chip: 'chipAlcohol',
        ic: 'i-wine',
        levels: { 'Ні': 'ok', 'Помірне вживання': 'warn', 'Часте вживання': 'bad' }
    },
    {
        select: 'lifeFamily',
        chip: 'chipFamily',
        ic: 'i-users',
        levels: { 'Не обтяжена': 'ok', 'Присутня (1-ша лінія)': 'bad' }
    },
];
const STATUS_ICON = { ok: 'i-check-circle', warn: 'i-alert', bad: 'i-alert', na: 'i-info' };

function renderLifeChips() {
    LIFE_FIELDS.forEach(f => {
        const chip = $(f.chip);
        const sel = $(f.select);
        if (!chip || !sel) return;
        const value = sel.value;
        const level = f.levels[value] || 'na';
        chip.className = `chip ${level} mt-1`;
        chip.innerHTML = value ?
            `${icon(f.ic, '')}${escapeHtml(value)}${icon(STATUS_ICON[level], 'ms-1')}` :
            `${icon('i-info', '')}Не вказано`;
    });
}

// Зведення виявлених факторів ризику (лабораторні + спосіб життя) з іконками
function renderRiskFactors(prof, life) {
    const box = $('riskFactors');
    if (!box) return;
    const d = isDiabetic();
    const items = [];
    const add = (level, ic, text) => items.push(`<span class="chip ${level}">${icon(ic, '')}${escapeHtml(text)}</span>`);

    if (isNum(prof.bmi)) {
        if (prof.bmi >= 30) add('bad', 'i-sliders', `Ожиріння (ІМТ ${fmt(prof.bmi)})`);
        else if (prof.bmi >= 25) add('warn', 'i-sliders', `Надмірна вага (ІМТ ${fmt(prof.bmi)})`);
    }
    if (isNum(prof.systolic_bp)) {
        if (prof.systolic_bp >= 140) add('bad', 'i-heart', `Гіпертензія (АТ ${fmt(prof.systolic_bp, 0)})`);
        else if (prof.systolic_bp >= 130) add('warn', 'i-heart', `АТ вище норми (${fmt(prof.systolic_bp, 0)})`);
    }
    if (isNum(prof.HbA1c)) {
        if (d) {
            if (prof.HbA1c >= 8) add('bad', 'i-droplet', `HbA1c вище цілі (${fmt(prof.HbA1c)}%)`);
            else if (prof.HbA1c >= 7) add('warn', 'i-droplet', `HbA1c вище цілі (${fmt(prof.HbA1c)}%)`);
        } else if (prof.HbA1c >= 6.5) add('bad', 'i-droplet', `HbA1c у діапазоні діабету (${fmt(prof.HbA1c)}%)`);
        else if (prof.HbA1c >= 5.7) add('warn', 'i-droplet', `Предіабет (HbA1c ${fmt(prof.HbA1c)}%)`);
    }
    if (isNum(prof.triglycerides) && prof.triglycerides >= 1.7) {
        add(prof.triglycerides >= 2.3 ? 'bad' : 'warn', 'i-droplet', `Тригліцериди ${fmt(prof.triglycerides)} ммоль/л`);
    }
    if (isNum(prof.age) && prof.age >= 45) add('warn', 'i-user', `Вік ${prof.age} р. (≥ 45)`);

    if (life.smoking === 'Активний курець') add('bad', 'i-smoke', 'Активне куріння');
    else if (life.smoking === 'Колишній курець') add('warn', 'i-smoke', 'Колишній курець');
    if (life.alcohol === 'Часте вживання') add('bad', 'i-wine', 'Часте вживання алкоголю');
    if (life.family_history === 'Присутня (1-ша лінія)') add('bad', 'i-users', 'Обтяжена спадковість');

    box.innerHTML = items.length ? items.join('') :
        `<span class="chip ok">${icon('i-check-circle', '')}Значущих факторів ризику не виявлено</span>`;
}

const TAB_IDS = { risk: 'tabRisk', complication: 'tabComplication', lab: 'tabLab' };
const PAGE_TITLES = { risk: 'Скринінг ризику діабету', complication: 'Ризик судинних ускладнень (ЦД 2 типу)', lab: 'Введення лабораторних даних' };

document.addEventListener('DOMContentLoaded', () => {
    setText('printDate', new Date().toLocaleDateString('uk-UA'));
    setText('topDate', new Date().toLocaleDateString('uk-UA', { day: 'numeric', month: 'long', year: 'numeric' }));
    const labForm = $('labForm');
    if (labForm) labForm.addEventListener('submit', handleLabSubmit);

    initCohort({ onOpen: openPatientFromCohort });

    // Прив'язка до window для доступу з HTML (onclick / onchange)
    Object.assign(window, {
        switchMode,
        switchChart,
        loadPatientData,
        saveLifestyle: saveLifestyleHandler,
        renderLifeChips,
        toggleCohort,
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

async function toggleCohort() {
    if (isCohortOpen()) {
        hideCohort();
        return;
    }
    setDisplay('mainDashboard', 'none');
    await openCohort(currentMode);
}

function hideCohort() {
    closeCohort();
    const select = $('patientSelect');
    if (select && select.value && globalPatientData) setDisplay('mainDashboard', 'block');
}

// Клік по пацієнту в таблиці/на графіку прогнозу → відкриваємо його картку
function openPatientFromCohort(id) {
    closeCohort();
    const search = $('patientSearch');
    if (search) search.value = id;
    loadPatientList(id); // точний збіг ID сам обирає пацієнта й завантажує картку
}

async function switchMode(mode) {
    currentMode = mode;
    closeCohort();
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

    applyNorms();

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
        const d = isDiabetic();

        setText('valAge', prof.age + ' р.');
        setText('valBmi', fmt(prof.bmi, 1, ' кг/м²'));
        setStatus('statusBmi', statusOf(STATUS.bmi, prof.bmi));
        setText('valGlucose', fmt(prof.glucose_fasting, 2, ' ммоль/л'));
        setStatus('statusGlucose', statusOf(STATUS.glucose, prof.glucose_fasting, d));
        setText('valBp', fmt(prof.systolic_bp, 0, ' мм рт.ст.'));
        setStatus('statusBp', statusOf(STATUS.bp, prof.systolic_bp));
        setText('valHba1cTile', fmt(prof.HbA1c, 1, '%'));
        setStatus('statusHba1c', statusOf(STATUS.hba1c, prof.HbA1c, d));
        setText('valTrig', fmt(prof.triglycerides, 2, ' ммоль/л'));
        setStatus('statusTrig', statusOf(STATUS.trig, prof.triglycerides));

        // Спосіб життя — реальні дані, які вводить лікар (не генеруються)
        setSelect('lifeSmoking', life.smoking);
        setSelect('lifeAlcohol', life.alcohol);
        setSelect('lifeFamily', life.family_history);
        setText('lifeMsg', '');
        renderLifeChips();
        renderRiskFactors(prof, life);

        setText('valHba1cCurrent', fmt(prof.HbA1c, 1, '%'));
        setText('dataQuality', (isNum(data.data_quality) ? data.data_quality : 100) + '%');
        $('patientInfoBadge').innerHTML =
            `<span class="badge bg-dark p-2 fs-6">ID: <b>${escapeHtml(data.patient_id)}</b> | Стать: <b>${escapeHtml(prof.sex)}</b> | Статус: <b>${data.is_diabetic ? 'Підтверджений ЦД2' : 'Без діабету'}</b></span>`;

        const comp = ml.complication;
        if (comp) { // Модуль 2: ризик ускладнень у відсотках (оцінка за правилами, не ML-ймовірність)
            setText('riskScore', comp.score + '%');
            setText('riskSubtitle', `Оцінка за правилами (не ML) · мікро ${comp.micro}% · макро ${comp.macro}% · прогноз: ${comp.forecast_score}%`);
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

        // Підсвічування відхилень: для ЦД2 — відносно цілей, для скринінгу — відносно діагностичних порогів
        const hbaLimit = d ? 7 : 6.5;
        const gluLimit = d ? 7.2 : 6.1;
        $('historyTable').querySelector('tbody').innerHTML = data.history.map(row => {
            const hba = isNum(row.HbA1c) ? Number(row.HbA1c) : null;
            const glu = isNum(row.glucose_fasting) ? Number(row.glucose_fasting) : null;
            const hbaBad = hba !== null && hba >= hbaLimit;
            const abnormal = hbaBad || (glu !== null && (d ? glu > gluLimit : glu >= gluLimit));
            return `
                <tr class="${abnormal ? 'table-danger' : ''}">
                    <td>${escapeHtml(row.formatted_date || row.test_date)}</td>
                    <td class="fw-bold ${hbaBad ? 'text-danger' : 'text-success'}">${fmt(hba, 1, '%')}</td>
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