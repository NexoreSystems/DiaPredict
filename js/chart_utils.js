const FORECAST_LABEL = 'ПРОГНОЗ';
let chart = null;
let lastData = null;
let current = 'hba1c';

const forecastOf = (data, key) =>
    data.trend_forecasts && data.trend_forecasts[key] != null ? data.trend_forecasts[key] : null;

// Один графік, три показники: перемикаються вкладками
const CHART_CONFIG = {
    hba1c: {
        field: 'HbA1c',
        label: 'HbA1c (%)',
        unit: '%',
        color: '#2b4585',
        fill: 'rgba(43, 69, 133, 0.14)',
        axisTitle: 'HbA1c (%)',
        forecastLabel: 'Прогноз HbA1c (ML Tier 2)',
        forecast: d => (d.ml_predictions ? d.ml_predictions.predicted_hba1c : null),
        refs: [{ value: 6.5, label: 'Поріг діабету (6.5%)', color: '#dc3545' },
            { value: 5.7, label: 'Предіабет (5.7%)', color: '#e0a800' }
        ]
    },
    glucose: {
        field: 'glucose_fasting',
        label: 'Глюкоза натще (ммоль/л)',
        unit: ' ммоль/л',
        color: '#fd7e14',
        fill: 'rgba(253, 126, 20, 0.12)',
        axisTitle: 'Глюкоза (ммоль/л)',
        forecastLabel: 'Прогноз глюкози (лінійний тренд)',
        forecast: d => forecastOf(d, 'glucose_fasting'),
        refs: [{ value: 7.0, label: 'Діабет (7.0)', color: '#dc3545' },
            { value: 5.6, label: 'Норма до 5.6', color: '#198754' }
        ]
    },
    bp: {
        field: 'systolic_bp',
        label: 'Систолічний АТ (мм рт.ст.)',
        unit: ' мм рт.ст.',
        color: '#dc3545',
        fill: 'rgba(220, 53, 69, 0.12)',
        axisTitle: 'АТ (мм рт.ст.)',
        forecastLabel: 'Прогноз АТ (лінійний тренд)',
        forecast: d => forecastOf(d, 'systolic_bp'),
        refs: [{ value: 140, label: 'Гіпертензія (140)', color: '#dc3545' },
            { value: 130, label: 'Норма до 130', color: '#198754' }
        ]
    }
};

function draw() {
    const canvas = document.getElementById('mainChart');
    if (!canvas || !lastData) return;
    const cfg = CHART_CONFIG[current];
    if (chart) chart.destroy();

    const history = lastData.history || [];
    const labels = history.map(h => h.formatted_date || h.test_date);
    const values = history.map(h => (h[cfg.field] === undefined ? null : h[cfg.field])); // пропуск = розрив лінії, а не нуль

    const lastIdx = values.reduce((acc, v, i) => (v !== null ? i : acc), -1);
    const forecast = lastIdx >= 0 ? cfg.forecast(lastData) : null;
    const hasForecast = forecast !== null && forecast !== undefined && !Number.isNaN(forecast);
    if (hasForecast) labels.push(FORECAST_LABEL);

    const datasets = [{
        label: cfg.label,
        data: values,
        borderColor: cfg.color,
        backgroundColor: cfg.fill,
        borderWidth: 3,
        tension: 0.3,
        fill: true,
        spanGaps: true,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: cfg.color,
        unit: cfg.unit
    }];

    if (hasForecast) {
        const fv = labels.map(() => null);
        fv[lastIdx] = values[lastIdx];
        fv[labels.length - 1] = forecast;
        datasets.push({
            label: cfg.forecastLabel,
            data: fv,
            borderColor: '#5b8def',
            borderDash: [6, 6],
            borderWidth: 3,
            spanGaps: true,
            pointRadius: 6,
            pointHoverRadius: 8,
            pointStyle: 'rectRot',
            pointBackgroundColor: '#5b8def',
            unit: cfg.unit
        });
    }

    cfg.refs.forEach(ref => datasets.push({
        label: ref.label,
        data: labels.map(() => ref.value),
        borderColor: ref.color,
        borderWidth: 1.5,
        borderDash: [3, 5],
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false,
        isRef: true
    }));

    chart = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
                tooltip: {
                    filter: item => !item.dataset.isRef && item.parsed.y !== null,
                    callbacks: { label: item => `${item.dataset.label}: ${item.parsed.y}${item.dataset.unit || ''}` }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, font: { size: 11 } } },
                y: { beginAtZero: false, grace: '10%', title: { display: true, text: cfg.axisTitle, font: { size: 11 } } }
            }
        }
    });
}

function markActiveTab() {
    document.querySelectorAll('[data-chart]').forEach(b =>
        b.classList.toggle('active', b.dataset.chart === current));
}

// Перемикання показника на графіку (HbA1c / глюкоза / АТ)
export function switchChart(key) {
    if (!CHART_CONFIG[key]) return;
    current = key;
    markActiveTab();
    draw();
}

export function renderCharts(patientData) {
    lastData = patientData;
    markActiveTab();
    draw();
}