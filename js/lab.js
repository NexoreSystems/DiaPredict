import { fetchAllLabPatients, saveLabRecord } from './api.js';

let allPatientsData = [];
let autoNextId = ''; // порожній, якщо база недоступна — щоб не видати дублікат ID

const $ = (id) => document.getElementById(id);
const todayLocal = () => new Date().toLocaleDateString('sv-SE'); // YYYY-MM-DD за локальним часом
const numberOr = (id, fallback) => parseFloat($(id).value || fallback);

function showAlert(type, text) {
    const box = $('labAlert');
    if (!box) return;
    box.style.display = 'block';
    box.className = `alert alert-${type} border-${type} fw-bold`;
    box.innerText = text;
}

// Ініціалізація лабораторного модуля (викликається при переході на вкладку)
export async function initLabModule() {
    const dateInput = $('labTestDate');
    if (dateInput && !dateInput.value) dateInput.value = todayLocal();

    try {
        const data = await fetchAllLabPatients();
        allPatientsData = data.patients || [];
        autoNextId = data.next_patient_id || '';
        populatePatientSelect();
    } catch (err) {
        console.error('Помилка завантаження бази лаборанта:', err);
        allPatientsData = [];
        autoNextId = '';
        const select = $('labPatientSelect');
        if (select) select.innerHTML = '<option value="">-- База недоступна --</option>';
        showAlert('danger', '❌ ' + err.message);
    }

    const modeNew = $('modeNew');
    handlePatientModeChange(modeNew && modeNew.checked ? 'new' : 'existing');
}

// Список існуючих пацієнтів: показуємо лише перші MAX_OPTIONS збігів за пошуком
const MAX_OPTIONS = 50;

function renderPatientOptions(query = '') {
    const select = $('labPatientSelect');
    if (!select) return;

    const q = query.trim().toLowerCase();
    const matches = q ? allPatientsData.filter(p => p.patient_id.toLowerCase().includes(q)) : allPatientsData;

    select.replaceChildren();
    matches.slice(0, MAX_OPTIONS).forEach(p => {
        const status = p.dx_type2_diabetes === 1 ? 'ЦД2' : 'Скринінг';
        select.add(new Option(`${p.patient_id} | Вік: ${p.age}, Стать: ${p.sex} (${status})`, p.patient_id));
    });

    const info = !matches.length ? 'Нічого не знайдено' :
        matches.length > MAX_OPTIONS ? `… ще ${matches.length - MAX_OPTIONS}. Уточніть пошук` : '';
    if (info) {
        const opt = new Option(info, '');
        opt.disabled = true;
        select.add(opt);
    }
}

function populatePatientSelect() {
    const search = $('labPatientSearch');
    if (search && !search.dataset.bound) {
        search.addEventListener('input', () => {
            renderPatientOptions(search.value);
            const exact = allPatientsData.find(p => p.patient_id.toLowerCase() === search.value.trim().toLowerCase());
            if (exact) {
                $('labPatientSelect').value = exact.patient_id;
                handleExistingPatientSelect();
            }
        });
        search.dataset.bound = '1';
    }
    renderPatientOptions(search ? search.value : '');
}

// Перемикання між існуючим та новим пацієнтом (автогенерація ID)
export function handlePatientModeChange(mode) {
    const idInput = $('labPatientId');
    if (!idInput) return;

    const isExisting = mode === 'existing';
    const existingGroup = $('labExistingGroup');
    const newGroup = $('labNewGroup');
    if (existingGroup) existingGroup.style.display = isExisting ? 'block' : 'none';
    if (newGroup) newGroup.style.display = isExisting ? 'none' : 'block';

    idInput.readOnly = true;
    idInput.value = isExisting ? '' : autoNextId;

    const age = $('labAge');
    if (age) {
        age.readOnly = isExisting;
        if (!isExisting) age.value = 45;
    }
    ['labSex', 'labDxDiabetes'].forEach(id => {
        const el = $(id);
        if (el) el.disabled = isExisting;
    });
}

// Автозаповнення полів при виборі існуючого пацієнта
export function handleExistingPatientSelect() {
    const select = $('labPatientSelect');
    if (!select) return;

    const patient = allPatientsData.find(p => p.patient_id === select.value);
    $('labPatientId').value = patient ? patient.patient_id : '';
    if (patient) {
        $('labAge').value = patient.age;
        $('labSex').value = patient.sex;
        $('labDxDiabetes').value = patient.dx_type2_diabetes;
    }
}

// Збереження нового лабораторного запису у базу через API
export async function handleLabSubmit(e) {
    e.preventDefault();

    const pid = $('labPatientId').value;
    if (!pid) {
        alert('Будь ласка, оберіть або згенеруйте ID пацієнта!');
        return;
    }

    const payload = {
        patient_mode: (document.querySelector('input[name="labPatientMode"]:checked') || {}).value || 'existing',
        patient_id: pid,
        test_date: $('labTestDate').value,
        dx_type2_diabetes: parseInt($('labDxDiabetes').value),
        age: parseInt($('labAge').value),
        sex: $('labSex').value,
        glucose_fasting_mmol: parseFloat($('labGlucose').value),
        hba1c: parseFloat($('labHba1c').value),
        bmi: parseFloat($('labBmi').value),
        systolic_bp: parseFloat($('labBpSystolic').value),
        diastolic_bp: numberOr('labBpDiastolic', 80),
        triglycerides: numberOr('labTriglycerides', 1.7),
        smoking: $('labSmoking').value || null,
        alcohol: $('labAlcohol').value || null,
        family_history: $('labFamily').value || null
    };

    const submitBtn = e.submitter || document.querySelector('#labForm [type="submit"]');
    if (submitBtn) submitBtn.disabled = true; // захист від подвійного натискання
    try {
        const { ok, data } = await saveLabRecord(payload);
        if (ok) {
            showAlert('success', '✅ ' + data.message);
            $('labForm').reset();
            $('labTestDate').value = todayLocal();
            await initLabModule(); // оновлюємо список і наступний згенерований ID
        } else {
            showAlert('danger', '❌ Помилка при збереженні аналізу: ' + data.message);
        }
    } finally {
        if (submitBtn) submitBtn.disabled = false;
    }
}