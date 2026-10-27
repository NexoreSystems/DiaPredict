const NETWORK_ERROR =
    "Немає зв'язку з сервером. Перевірте, що FastAPI запущено і сторінка відкрита " +
    "через його адресу (напр. http://127.0.0.1:8000), а не через Live Server.";

// Текст помилки з відповіді FastAPI (detail може бути рядком або масивом 422)
function extractDetail(data, status) {
    const detail = data && data.detail;
    if (Array.isArray(detail)) {
        return detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    }
    return detail || `Помилка сервера (${status})`;
}

async function request(url, options) {
    let res;
    try {
        res = await fetch(url, options);
    } catch (e) {
        throw new Error(NETWORK_ERROR);
    }
    const data = await res.json().catch(() => null); // відповідь може бути не JSON
    return { ok: res.ok, status: res.status, data };
}

async function getJson(url, errorPrefix) {
    const { ok, status, data } = await request(url);
    if (!ok) throw new Error(`${errorPrefix}: ${extractDetail(data, status)}`);
    return data;
}

export const fetchPatients = (mode, query = '') =>
    getJson(`/api/patients?mode=${encodeURIComponent(mode)}&q=${encodeURIComponent(query)}`,
        'Помилка завантаження списку пацієнтів');

export const fetchAllLabPatients = () =>
    getJson('/api/patients/all', 'Помилка завантаження бази для лаборанта');

export const fetchPatientHistory = (patientId, mode) =>
    getJson(`/api/patient/${encodeURIComponent(patientId)}/history?mode=${encodeURIComponent(mode)}`,
        'Помилка завантаження детальної картки');

// Повертає { ok, data }; при помилці data.message містить причину
export async function saveLabRecord(payload) {
    try {
        const { ok, status, data } = await request('/api/patient/add_record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return { ok, data: ok ? data : { message: extractDetail(data, status) } };
    } catch (err) {
        return { ok: false, data: { message: err.message } };
    }
}

// Збереження способу життя (куріння, алкоголь, спадковість), які вводить лікар
export async function saveLifestyle(patientId, payload) {
    try {
        const { ok, status, data } = await request(`/api/patient/${encodeURIComponent(patientId)}/lifestyle`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return { ok, data: ok ? data : { message: extractDetail(data, status) } };
    } catch (err) {
        return { ok: false, data: { message: err.message } };
    }
}