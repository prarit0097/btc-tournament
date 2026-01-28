let scoreboardCache = [];
let lastPrediction = null;
let lastNowPrice = null;
let lastNowPriceInr = null;
let lastFxRate = null;
let lastFxUpdatedAt = null;
let lastFxSource = null;
let lastForcedRefreshAt = 0;

async function getJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function fmt(num, digits = 2) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return Number(num).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function fmtUsd(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return `$${fmt(num, 2)}`;
}

function fmtInr(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return `INR ${fmt(num, 2)}`;
}

function formatDualPrice(usd, fxRate) {
  if (usd === null || usd === undefined || Number.isNaN(usd)) return '--';
  if (fxRate) {
    return `${fmtInr(usd * fxRate)} (${fmtUsd(usd)})`;
  }
  return fmtUsd(usd);
}

function formatNowPrice(usd, inr, fxRate) {
  if (inr !== null && inr !== undefined && !Number.isNaN(inr)) {
    return `${fmtInr(inr)}${usd ? ` (${fmtUsd(usd)})` : ''}`;
  }
  return formatDualPrice(usd, fxRate);
}

function fmtDateTime(iso) {
  if (!iso) return '--';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

function fmtDateTimeLower(iso) {
  const text = fmtDateTime(iso);
  if (text === '--') return text;
  return text.replace(' AM', ' am').replace(' PM', ' pm');
}

function normalizePredictions(data) {
  if (!data || !Array.isArray(data.predictions)) return [];
  return data.predictions;
}

function predictionMinutes(pred) {
  if (!pred) return 0;
  return pred.prediction_horizon_min || pred.timeframe_minutes || 0;
}

function labelForPrediction(pred) {
  if (!pred) return '--';
  if (pred.timeframe) return pred.timeframe;
  const mins = predictionMinutes(pred);
  if (!mins) return '--';
  return mins < 60 ? `${mins}m` : `${mins / 60}h`;
}

function sortPredictions(preds) {
  return [...preds].sort((a, b) => predictionMinutes(a) - predictionMinutes(b));
}

function selectPrimaryPrediction(preds) {
  if (!preds.length) return null;
  const byMinutes = sortPredictions(preds);
  const ten = byMinutes.find(p => predictionMinutes(p) === 10) || byMinutes[0];
  return ten || byMinutes[0];
}

function formatCountdown(predictedAt, horizonMin) {
  if (!predictedAt) return '--:--';
  const start = new Date(predictedAt).getTime();
  if (Number.isNaN(start)) return '--:--';
  const target = start + horizonMin * 60 * 1000;
  const diff = target - Date.now();
  if (diff <= 0) return '00:00';
  const totalSec = Math.floor(diff / 1000);
  const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
  const ss = String(totalSec % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

function statusText(pred) {
  if (!pred) return '--';
  if (pred.match_percent !== null && pred.match_percent !== undefined) {
    return `${fmt(pred.match_percent, 1)}%`;
  }
  if (pred.status === 'pending') {
    const horizonMin = predictionMinutes(pred);
    return `Pending (in ${formatCountdown(pred.predicted_at, horizonMin)})`;
  }
  if (pred.status && pred.status !== 'ready') {
    return pred.status.replace('_', ' ');
  }
  return '--';
}

function renderPriceRow(primary, nowPriceUsd, nowPriceInr) {
  const nowDisplay = formatNowPrice(nowPriceUsd, nowPriceInr, lastFxRate);
  const predList = document.getElementById('pred-list');
  const lastLine = document.getElementById('pred-last-line');
  const actualLine = document.getElementById('pred-actual-line');

  if (!primary) {
    document.getElementById('price-now').textContent = `BTC Now: ${nowDisplay}`;
    document.getElementById('price-row').innerHTML = '<span class="price-left">Predicted: -- | Match: --</span><span class="price-right"></span>';
    if (lastLine) lastLine.textContent = 'Last predicted price: --';
    if (actualLine) actualLine.textContent = 'Actual price at match time: --';
    return;
  }

  const horizonMin = predictionMinutes(primary) || 10;
  const label = labelForPrediction(primary);
  const predDisplay = primary.predicted_price
    ? formatDualPrice(primary.predicted_price, lastFxRate)
    : '--';
  const match = statusText(primary);
  const isSingle = Array.isArray(lastPrediction?.predictions) && lastPrediction.predictions.length <= 1;

  const lastReady = primary.last_ready;
  let lastBlock = '';
  if (lastReady) {
    const lastLineText = `Last predicted price: ${formatDualPrice(lastReady.predicted_price, lastFxRate)} (${fmt(lastReady.match_percent, 1)}%)`;
    const actualLineText = lastReady.actual_price !== null && lastReady.actual_price !== undefined
      ? `Actual price at match time: ${formatDualPrice(lastReady.actual_price, lastFxRate)}`
      : 'Actual price at match time: --';
    const actualTime = lastReady.actual_at ? `at ${fmtDateTimeLower(lastReady.actual_at)}` : '';
    lastBlock = `${lastLineText}\n${actualLineText}${actualTime ? ' ' + actualTime : ''}`;
  }

  if (isSingle) {
    document.getElementById('price-now').textContent = `BTC Now: ${nowDisplay}`;
    document.getElementById('price-row').innerHTML = `
      <span class="price-left">Predicted (${label || `${horizonMin}m`}): ${predDisplay} | Match: ${match}</span>
      <span class="price-right"></span>
    `;
    if (lastReady) {
      const actualTime = lastReady.actual_at ? `at ${fmtDateTimeLower(lastReady.actual_at)}` : '';
      if (lastLine) {
        lastLine.textContent = `Last predicted price: ${formatDualPrice(lastReady.predicted_price, lastFxRate)} (${fmt(lastReady.match_percent, 1)}%)`;
      }
      if (actualLine) {
        const actualText = lastReady.actual_price !== null && lastReady.actual_price !== undefined
          ? `Actual price at match time: ${formatDualPrice(lastReady.actual_price, lastFxRate)}`
          : 'Actual price at match time: --';
        actualLine.textContent = `${actualText}${actualTime ? ' ' + actualTime : ''}`;
      }
    } else {
      if (lastLine) lastLine.textContent = 'Last predicted price: --';
      if (actualLine) actualLine.textContent = 'Actual price at match time: --';
    }
    if (predList) predList.style.display = 'none';
  } else {
    document.getElementById('price-now').textContent = `BTC Now: ${nowDisplay}`;
    document.getElementById('price-row').innerHTML = `
      <span class="price-left">Predicted (${label || `${horizonMin}m`}): ${predDisplay} | Match: ${match}</span>
      <span class="price-right">${lastBlock ? lastBlock.replace('\\n', ' | ') : ''}</span>
    `;
    if (lastLine) lastLine.textContent = 'Last predicted price: --';
    if (actualLine) actualLine.textContent = 'Actual price at match time: --';
    if (predList) predList.style.display = '';
  }
}

function renderPredList(predictions) {
  const list = document.getElementById('pred-list');
  if (!list) return;
  if (predictions.length <= 1) {
    list.innerHTML = '';
    list.style.display = 'none';
    return;
  }
  if (!predictions.length) {
    list.innerHTML = '';
    list.style.display = 'none';
    return;
  }
  list.style.display = '';
  const ordered = sortPredictions(predictions);
  list.innerHTML = '';
  ordered.forEach((pred, idx) => {
    const label = labelForPrediction(pred);
    const horizonMin = predictionMinutes(pred) || 0;
    const predPrice = pred.predicted_price
      ? formatDualPrice(pred.predicted_price, lastFxRate)
      : '--';
    const match = statusText(pred);

    let lastBlock = 'Last predicted: -- | Actual: --';
    if (pred.last_ready) {
      const lr = pred.last_ready;
      const lastLine = `Last predicted: ${formatDualPrice(lr.predicted_price, lastFxRate)} (${fmt(lr.match_percent, 1)}%)`;
      const actualLine = lr.actual_price !== null && lr.actual_price !== undefined
        ? `Actual: ${formatDualPrice(lr.actual_price, lastFxRate)}`
        : 'Actual: --';
      const actualTime = lr.actual_at ? `@ ${fmtDateTimeLower(lr.actual_at)}` : '';
      lastBlock = `${lastLine} | ${actualLine}${actualTime ? ' ' + actualTime : ''}`;
    }

    const line = `Predicted (${label}): ${predPrice} | Match: ${match} | ${lastBlock}`;

    const item = document.createElement('div');
    item.className = 'pred-item';
    item.innerHTML = `<div class="pred-idx">${idx + 1}</div><div class="pred-text">${line}</div>`;
    list.appendChild(item);
  });
}

function updateTimeframePill(predictions) {
  const pill = document.getElementById('timeframe-pill');
  if (!pill) return;
  if (!predictions || predictions.length === 0) {
    pill.textContent = 'timeframe: --';
    return;
  }
  if (predictions.length === 1) {
    pill.textContent = `${labelForPrediction(predictions[0])} candles`;
  } else {
    pill.textContent = 'multi-timeframe';
  }
}

function renderPredictionUI() {
  const predictions = normalizePredictions(lastPrediction);
  updateTimeframePill(predictions);
  const primary = selectPrimaryPrediction(predictions.filter(p => p.predicted_price));
  renderPriceRow(primary, lastNowPrice, lastNowPriceInr);
  renderPredList(predictions);
}

async function loadPrice() {
  try {
    const data = await getJSON('/api/btc/price');
    const updated = new Date(data.updated_at).toLocaleTimeString();
    let fxText = '';
    if (data.fx_rate) {
      fxText = ` | FX: 1 USD = ${fmt(data.fx_rate, 2)} INR`;
      if (data.fx_source) fxText += ` (${data.fx_source})`;
      if (data.fx_stale) fxText += ' stale';
    }
    document.getElementById('price-updated').textContent = `Updated: ${updated}${fxText}`;
    lastNowPrice = data.price;
    lastNowPriceInr = data.price_inr;
    if (data.fx_rate) lastFxRate = data.fx_rate;
    if (data.fx_updated_at) lastFxUpdatedAt = data.fx_updated_at;
    if (data.fx_source) lastFxSource = data.fx_source;
    renderPredictionUI();
    return data.price;
  } catch (err) {
    return null;
  }
}

async function loadSummary() {
  try {
    const data = await getJSON('/api/btc/tournament/summary');
    document.getElementById('candidate-count').textContent = `${data.candidate_count || 0} models`;
    document.getElementById('last-run').textContent = `Last run: ${fmtDateTime(data.last_run_at)}`;
    document.getElementById('last-completed').textContent = `Last tournament completed: ${fmtDateTimeLower(data.last_run_at)}`;
    document.getElementById('run-mode').textContent = `mode: ${data.run_mode || '--'}`;

    const champs = data.champions || {};
    document.getElementById('champ-direction').textContent = `Direction champion: ${champs.direction?.model_id || '--'}`;
    document.getElementById('champ-return').textContent = `Return champion: ${champs.return?.model_id || '--'}`;
    document.getElementById('champ-range').textContent = `Range champion: ${champs.range?.model_id || '--'}`;
  } catch (err) {
    // ignore
  }
}

function renderScoreboard(rows) {
  const tbody = document.getElementById('scoreboard-body');
  tbody.innerHTML = '';
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    if (row.is_champion) tr.classList.add('winner');
    const badge = row.is_champion ? ' <span class="badge">Champion</span>' : '';
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.target}</td>
      <td>${row.feature_set}</td>
      <td>${row.model_name}${badge}</td>
      <td>${row.family}</td>
      <td>${fmt(row.final_score, 4)}</td>
      <td>${row.primary_metric?.name || ''}: ${fmt(row.primary_metric?.value, 4)}</td>
      <td>${fmt(row.trading_score, 4)}</td>
      <td>${fmt(row.stability_penalty, 4)}</td>
      <td>${row.run_at || '--'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderChart(rows) {
  const chart = document.getElementById('score-chart');
  chart.innerHTML = '';
  const top = rows.slice(0, 10);
  const maxScore = Math.max(...top.map(r => r.final_score || 0), 1e-6);
  top.forEach((r) => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = `${Math.max(5, (r.final_score / maxScore) * 100)}%`;
    bar.innerHTML = `<span>${fmt(r.final_score, 3)}</span>`;
    chart.appendChild(bar);
  });
}

function applyFilters() {
  let rows = [...scoreboardCache];
  const target = document.getElementById('filter-target').value;
  const feature = document.getElementById('filter-feature').value;
  const text = document.getElementById('filter-text').value.toLowerCase();
  const sortBy = document.getElementById('sort-by').value;

  if (target !== 'all') rows = rows.filter(r => r.target === target);
  if (feature !== 'all') rows = rows.filter(r => r.feature_set === feature);
  if (text) rows = rows.filter(r => (r.model_name || '').toLowerCase().includes(text));

  rows.sort((a, b) => {
    if (sortBy === 'trading_score') return (b.trading_score || 0) - (a.trading_score || 0);
    if (sortBy === 'primary') return (b.primary_metric?.value || 0) - (a.primary_metric?.value || 0);
    return (b.final_score || 0) - (a.final_score || 0);
  });

  renderScoreboard(rows);
  renderChart(rows);
}

async function loadScoreboard() {
  try {
    const rows = await getJSON('/api/btc/tournament/scoreboard?limit=500');
    scoreboardCache = rows;
    applyFilters();
  } catch (err) {
    // ignore
  }
}

async function refreshPrediction() {
  try {
    await getJSON('/api/btc/prediction/refresh', { method: 'POST' });
  } catch (err) {
    // ignore
  }
}

async function loadPrediction() {
  try {
    const data = await getJSON('/api/btc/prediction/latest');
    const nowPrice = await loadPrice();
    lastPrediction = data;
    if (nowPrice !== null && nowPrice !== undefined) {
      lastNowPrice = nowPrice;
    }
    renderPredictionUI();
  } catch (err) {
    // ignore
  }
}

async function runNow() {
  const button = document.getElementById('run-now');
  const state = document.getElementById('run-state');
  button.disabled = true;
  state.textContent = 'running...';
  try {
    const res = await getJSON('/api/btc/tournament/run', { method: 'POST', body: '{}' });
    state.textContent = res.status || 'started';
    if (!res.running) {
      button.disabled = false;
    }
  } catch (err) {
    state.textContent = 'error';
  }
}

async function pollRunStatus() {
  try {
    const state = await getJSON('/api/btc/tournament/run/status');
    const badge = document.getElementById('run-state');
    if (state.running) {
      badge.textContent = 'running';
      document.getElementById('run-now').disabled = true;
    } else {
      badge.textContent = 'idle';
      document.getElementById('run-now').disabled = false;
    }
  } catch (err) {
    // ignore
  }
}

function updateCountdownOnly() {
  const predictions = normalizePredictions(lastPrediction);
  if (predictions.length) {
    const now = Date.now();
    const due = predictions.some((pred) => {
      if (!pred || pred.status !== 'pending' || !pred.predicted_at) return false;
      const horizonMin = predictionMinutes(pred) || 0;
      const start = new Date(pred.predicted_at).getTime();
      if (Number.isNaN(start)) return false;
      return now >= start + horizonMin * 60 * 1000;
    });
    if (due && now - lastForcedRefreshAt > 5000) {
      lastForcedRefreshAt = now;
      loadPrediction();
      return;
    }
  }
  renderPredictionUI();
}

async function init() {
  await loadSummary();
  await loadScoreboard();
  await refreshPrediction();
  await loadPrediction();

  document.getElementById('run-now').addEventListener('click', runNow);
  document.getElementById('filter-target').addEventListener('change', applyFilters);
  document.getElementById('filter-feature').addEventListener('change', applyFilters);
  document.getElementById('filter-text').addEventListener('input', applyFilters);
  document.getElementById('sort-by').addEventListener('change', applyFilters);

  setInterval(loadPrice, 15000);
  setInterval(loadPrediction, 15000);
  setInterval(refreshPrediction, 15000);
  setInterval(loadSummary, 60000);
  setInterval(loadScoreboard, 60000);
  setInterval(pollRunStatus, 10000);
  setInterval(updateCountdownOnly, 1000);
}

init();
