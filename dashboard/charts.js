/**
 * charts.js — Pure chart-drawing functions for AgentAssert Dashboard.
 *
 * All functions are side-effect-free beyond writing to the canvas/SVG
 * element passed in. No global state. No external dependencies.
 *
 * Exports (attached to window.Charts for consumption by app.js):
 *   drawEProcessChart(canvas, certData)
 *   drawCompSvg(svgEl, compData)
 *   buildCIBar(container, depData)    — returns a DOM fragment
 *   buildBreakdownBars(container, breakdownMap, label)
 */

(function () {
  'use strict';

  // ── Canvas setup with device-pixel-ratio scaling ─────────────────
  function setupCanvas(canvas, cssWidth, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(cssWidth  * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width  = cssWidth  + 'px';
    canvas.style.height = cssHeight + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return ctx;
  }

  // ── Palette ───────────────────────────────────────────────────────
  const C = {
    bg:       '#0f172a',
    axis:     '#334155',
    textDim:  '#94a3b8',
    text:     '#f1f5f9',
    blue:     '#3b82f6',
    green:    '#22c55e',
    red:      '#ef4444',
    yellow:   '#eab308',
    threshold:'#ef4444',
  };

  // ── Numeric formatting ────────────────────────────────────────────
  function fmt4(n) { return (typeof n === 'number') ? n.toFixed(4) : '—'; }
  function fmt2(n) { return (typeof n === 'number') ? n.toFixed(2)  : '—'; }

  // ── drawEProcessChart ─────────────────────────────────────────────
  /**
   * Draw the log-wealth e-process curve on `canvas`.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {Object} certData  — certification payload from data.js
   *   wealth_curve           : number[]  (log-wealth per mission)
   *   threshold              : number    (log(1/alpha))
   *   first_crossing_index   : number|null
   *   certified              : boolean
   *   n_missions             : number
   */
  function drawEProcessChart(canvas, certData) {
    const { wealth_curve, threshold, first_crossing_index, certified } = certData;
    if (!wealth_curve || wealth_curve.length === 0) return;

    const cssW = canvas.clientWidth  || 760;
    const cssH = canvas.clientHeight || 260;
    const ctx = setupCanvas(canvas, cssW, cssH);

    const PAD = { top: 18, right: 24, bottom: 38, left: 52 };
    const W = cssW - PAD.left - PAD.right;
    const H = cssH - PAD.top  - PAD.bottom;

    const n = wealth_curve.length;
    const yMin = Math.min(...wealth_curve, 0) - 0.3;
    const yMax = Math.max(...wealth_curve, threshold) + 0.5;
    const xScale = x  => PAD.left + (x / (n - 1 || 1)) * W;
    const yScale = y  => PAD.top  + (1 - (y - yMin) / (yMax - yMin)) * H;

    // Background
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, cssW, cssH);

    // Grid lines (horizontal)
    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 0.5;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const yVal = yMin + (i / yTicks) * (yMax - yMin);
      const py = yScale(yVal);
      ctx.beginPath();
      ctx.moveTo(PAD.left, py);
      ctx.lineTo(PAD.left + W, py);
      ctx.stroke();
      ctx.fillStyle = C.textDim;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(fmt2(yVal), PAD.left - 4, py + 3.5);
    }

    // X-axis tick labels (every ~50 missions)
    const step = Math.max(1, Math.round(n / 8));
    ctx.fillStyle = C.textDim;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i += step) {
      const px = xScale(i);
      ctx.fillText(i + 1, px, cssH - PAD.bottom + 14);
    }
    // Always label last point
    ctx.fillText(n, xScale(n - 1), cssH - PAD.bottom + 14);

    // Axis labels
    ctx.fillStyle = C.textDim;
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Mission', PAD.left + W / 2, cssH - 4);
    ctx.save();
    ctx.translate(12, PAD.top + H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('log E(r)', 0, 0);
    ctx.restore();

    // Threshold line (red dashed)
    const pyThresh = yScale(threshold);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = C.threshold;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, pyThresh);
    ctx.lineTo(PAD.left + W, pyThresh);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.threshold;
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('log(1/α) = ' + fmt2(threshold), PAD.left + 4, pyThresh - 4);

    // Wealth curve (blue)
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = xScale(i);
      const py = yScale(wealth_curve[i]);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // First-crossing marker (green vertical line + dot)
    if (certified && first_crossing_index != null) {
      const xi = first_crossing_index - 1;   // 1-based → 0-based index
      const px = xScale(xi);
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, PAD.top);
      ctx.lineTo(px, PAD.top + H);
      ctx.stroke();
      ctx.setLineDash([]);
      const py = yScale(wealth_curve[xi]);
      ctx.fillStyle = C.green;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = C.green;
      ctx.font = '9px sans-serif';
      ctx.textAlign = px > PAD.left + W * 0.7 ? 'right' : 'left';
      ctx.fillText('certified r=' + first_crossing_index, px + (px > PAD.left + W * 0.7 ? -6 : 6), PAD.top + 12);
    }

    // Axes (draw on top)
    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, PAD.top + H);
    ctx.lineTo(PAD.left + W, PAD.top + H);
    ctx.stroke();
  }

  // ── drawCompSvg ───────────────────────────────────────────────────
  /**
   * Draw observed_reliability and independence_product as horizontal
   * bars in the provided <svg> element.
   */
  function drawCompSvg(svgEl, compData) {
    const { observed_reliability, independence_product } = compData;
    const W = svgEl.clientWidth || 380;
    const LW = 155;  // label column width
    const BW = W - LW - 70;  // bar area
    const rows = [
      { label: 'Observed Θ',       val: observed_reliability, col: '#3b82f6' },
      { label: 'Indep. product',   val: independence_product, col: '#64748b' },
    ];
    const rowH = 36;
    const svgH = rows.length * rowH + 20;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${svgH}`);
    svgEl.setAttribute('height', svgH);

    let html = '';
    rows.forEach((row, i) => {
      const y = i * rowH + 12;
      const barW = Math.max(0, row.val * BW);
      const pct   = (row.val * 100).toFixed(2) + '%';
      html += `
        <text x="${LW - 6}" y="${y + 10}" text-anchor="end"
              font-size="11" fill="#94a3b8">${esc(row.label)}</text>
        <rect x="${LW}" y="${y}" width="${BW}" height="14" rx="3"
              fill="#0f172a"/>
        <rect x="${LW}" y="${y}" width="${barW.toFixed(1)}" height="14" rx="3"
              fill="${row.col}"/>
        <text x="${LW + BW + 6}" y="${y + 10}"
              font-size="11" fill="#f1f5f9">${pct}</text>`;
    });
    svgEl.innerHTML = html;
  }

  // ── buildCIBar ────────────────────────────────────────────────────
  /**
   * Build the τ_a confidence-interval bar DOM fragment.
   * Returns a DocumentFragment ready to append.
   */
  function buildCIBar(container, depData) {
    if (depData.error) return;
    const { tau_a, tau_a_ci } = depData;
    const { lower, upper } = tau_a_ci;
    const RANGE = 2;  // [-1, 1]
    const toP = v => ((v + 1) / RANGE * 100).toFixed(2) + '%';

    const wrap = document.createElement('div');
    wrap.className = 'tau-ci-section';

    const label = document.createElement('div');
    label.className = 'tau-ci-label';
    label.textContent = 'τ_a = ' + fmt4(tau_a) + '  |  95% CI [' + fmt4(lower) + ', ' + fmt4(upper) + ']';
    wrap.appendChild(label);

    const barWrap = document.createElement('div');
    barWrap.className = 'tau-ci-bar-wrap';

    // CI range
    const rangeEl = document.createElement('div');
    rangeEl.className = 'tau-ci-range';
    rangeEl.style.left  = toP(Math.max(-1, lower));
    rangeEl.style.width = ((Math.min(1, upper) - Math.max(-1, lower)) / RANGE * 100).toFixed(2) + '%';
    barWrap.appendChild(rangeEl);

    // Zero line
    const zeroEl = document.createElement('div');
    zeroEl.className = 'tau-ci-zero';
    zeroEl.style.left = toP(0);
    barWrap.appendChild(zeroEl);

    // Point estimate
    const pointEl = document.createElement('div');
    pointEl.className = 'tau-ci-point';
    pointEl.style.left = toP(tau_a);
    barWrap.appendChild(pointEl);

    wrap.appendChild(barWrap);

    const axis = document.createElement('div');
    axis.className = 'tau-axis';
    axis.innerHTML = '<span>−1</span><span>0</span><span>+1</span>';
    wrap.appendChild(axis);

    container.appendChild(wrap);
  }

  // ── HTML escaping (XSS guard) ─────────────────────────────────────
  function esc(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Public API ────────────────────────────────────────────────────
  window.Charts = { drawEProcessChart, drawCompSvg, buildCIBar, esc };
})();
