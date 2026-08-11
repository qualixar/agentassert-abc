/**
 * app.js — AgentAssert Dashboard main application.
 *
 * Reads window.DASHBOARD_DATA (written by export_dashboard_data.py → data.js)
 * and renders all dashboard sections. No external dependencies.
 *
 * Section render functions:
 *   renderKPIs(data)
 *   renderCertification(cert)
 *   renderComposition(comp)
 *   renderDependence(dep)
 *   renderDrift(drift)
 *   renderBreakdown(motifMap, condMap)
 */

(function () {
  'use strict';

  const esc = (window.Charts && window.Charts.esc) || (s => String(s));

  // ── DOM helpers ───────────────────────────────────────────────────
  function el(id)   { return document.getElementById(id); }
  function chip(label, value, cls) {
    return `<div class="stat-chip">${esc(label)}: <span class="${esc(cls || '')}">${esc(value)}</span></div>`;
  }
  function badge(text, cls) {
    return `<span class="badge ${esc(cls)}">${esc(text)}</span>`;
  }
  function fmt(n, d) {
    if (n == null) return '—';
    return typeof n === 'number' ? n.toFixed(d ?? 4) : String(n);
  }

  // ── KPI cards ─────────────────────────────────────────────────────
  function renderKPIs(data) {
    const { meta, composition, certification, dependence } = data;
    const certified = certification.certified;
    const tau = (dependence && !dependence.error) ? dependence.tau_a : null;

    const cards = [
      {
        label: 'Missions',
        value: meta.n_missions,
        sub:   'logged',
        cls:   'kpi-blue',
      },
      {
        label: 'Reliability Θ',
        value: (composition.observed_reliability * 100).toFixed(2) + '%',
        sub:   'observed',
        cls:   composition.observed_reliability >= 0.90 ? 'kpi-green' : 'kpi-red',
      },
      {
        label: 'Certified',
        value: certified ? 'YES' : 'NO',
        sub:   'e-process p₀=' + (meta.p0 ?? 0.9) + '  α=' + (meta.alpha ?? 0.05),
        cls:   certified ? 'kpi-green' : 'kpi-red',
      },
      {
        label: 'τ_a (dependence)',
        value: tau != null ? fmt(tau) : '—',
        sub:   'Kendall binary',
        cls:   tau != null && Math.abs(tau) < 0.1 ? 'kpi-green' : 'kpi-yellow',
      },
      {
        label: 'Budget',
        value: '$0.00',
        sub:   'dry-run / local',
        cls:   'kpi-green',
      },
    ];

    el('kpi-row').innerHTML = cards.map(c => `
      <div class="kpi-card">
        <div class="kpi-label">${esc(c.label)}</div>
        <div class="kpi-value ${esc(c.cls)}">${esc(c.value)}</div>
        <div class="kpi-sub">${esc(c.sub)}</div>
      </div>`).join('');
  }

  // ── Certification ─────────────────────────────────────────────────
  function renderCertification(cert) {
    // Badge next to section title
    const badgeEl = el('cert-badge-inline');
    if (cert.certified) {
      badgeEl.textContent = 'CERTIFIED';
      badgeEl.className = 'badge badge-green';
    } else {
      badgeEl.textContent = 'NOT CERTIFIED';
      badgeEl.className = 'badge badge-red';
    }

    // Header badge
    const hb = el('header-badge');
    hb.textContent = cert.certified ? 'CERTIFIED' : 'NOT CERTIFIED';
    hb.className = 'cert-badge badge ' + (cert.certified ? 'badge-green' : 'badge-red');

    // Canvas chart
    const canvas = el('eprocess-canvas');
    if (canvas && window.Charts) {
      // Allow CSS to size the canvas first
      requestAnimationFrame(() => {
        Charts.drawEProcessChart(canvas, cert);
      });
    }

    // Stats
    el('cert-stats').innerHTML = [
      chip('Final wealth E',      fmt(cert.final_wealth, 2)),
      chip('First crossing',      cert.first_crossing_index != null
                                    ? 'mission ' + cert.first_crossing_index
                                    : 'never'),
      chip('Threshold 1/α',       (1 / cert.alpha).toFixed(0)),
      chip('log(1/α)',            fmt(cert.threshold, 4)),
      chip('Missions',            cert.n_missions),
      chip('p₀',                  cert.p0),
      chip('α',                   cert.alpha),
    ].join('');
  }

  // ── Composition ───────────────────────────────────────────────────
  function renderComposition(comp) {
    const svgEl = el('comp-svg');
    if (svgEl && window.Charts) {
      requestAnimationFrame(() => Charts.drawCompSvg(svgEl, comp));
    }

    const gapCls = comp.gap <= 0 ? 'kpi-green' : 'kpi-red';
    el('comp-stats').innerHTML = [
      chip('Observed Θ',     (comp.observed_reliability * 100).toFixed(4) + '%'),
      chip('Indep. product', (comp.independence_product * 100).toFixed(4) + '%'),
      chip('Gap',            fmt(comp.gap), gapCls),
      chip('Components',     comp.n_components),
      chip('Handoffs',       comp.n_handoffs),
      chip('Missions',       comp.n_missions),
    ].join('');
  }

  // ── Dependence ────────────────────────────────────────────────────
  function renderDependence(dep) {
    const tableWrap = el('cofailure-table-wrap');
    const statsWrap = el('dep-stats');

    if (dep.error) {
      tableWrap.innerHTML = `<p class="stat-chip">${esc(dep.error)}</p>`;
      return;
    }

    const { table, agent_pair, tau_a, tetrachoric_rho, n_missions } = dep;
    const [ai, aj] = agent_pair || ['A', 'B'];

    // 2×2 table — use textContent on individual cells (XSS-safe)
    const tbl = document.createElement('table');
    tbl.className = 'cofail-table';
    tbl.innerHTML = `
      <tr>
        <th></th>
        <th>${esc(aj)} fails</th>
        <th>${esc(aj)} passes</th>
      </tr>
      <tr>
        <th>${esc(ai)} fails</th>
        <td class="cell-n11"></td>
        <td class="cell-n10"></td>
      </tr>
      <tr>
        <th>${esc(ai)} passes</th>
        <td class="cell-n01"></td>
        <td class="cell-n00"></td>
      </tr>`;

    // Set cell text with textContent (never innerHTML) — XSS guard
    tbl.querySelector('.cell-n11').textContent = table.n11;
    tbl.querySelector('.cell-n10').textContent = table.n10;
    tbl.querySelector('.cell-n01').textContent = table.n01;
    tbl.querySelector('.cell-n00').textContent = table.n00;
    tableWrap.appendChild(tbl);

    // CI bar
    if (window.Charts) {
      Charts.buildCIBar(el('tau-ci-wrap'), dep);
    }

    statsWrap.innerHTML = [
      chip('Agent pair', ai + ' × ' + aj),
      chip('τ_a',        fmt(tau_a)),
      chip('ρ tetrachoric', tetrachoric_rho != null ? fmt(tetrachoric_rho) : 'N/A (degenerate)'),
      chip('Missions',   n_missions),
      chip('n₁₁ (both fail)', table.n11),
      chip('n₀₀ (both pass)', table.n00),
    ].join('');
  }

  // ── Drift ─────────────────────────────────────────────────────────
  function renderDrift(drift) {
    const { n_agents, n_passing, n_failing_gate, n_fit_error, agent_results } = drift;

    // Summary bar
    el('drift-summary-row').innerHTML = [
      chip('Total agents',  n_agents),
      chip('Gate passed',   n_passing,      n_passing     ? 'kpi-green'  : ''),
      chip('Gate failed',   n_failing_gate, n_failing_gate ? 'kpi-red'   : ''),
      chip('Fit error',     n_fit_error,    n_fit_error   ? 'kpi-yellow' : ''),
    ].join('');

    // Per-agent table
    const wrap = el('drift-agents-table');
    if (!agent_results || agent_results.length === 0) {
      wrap.innerHTML = '<p class="stat-chip">No agent results.</p>';
      return;
    }

    const tbl = document.createElement('table');
    tbl.className = 'drift-table';
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Agent ID</th><th>Obs</th><th>Gate</th><th>Note</th></tr>';
    tbl.appendChild(thead);

    const tbody = document.createElement('tbody');
    agent_results.forEach(r => {
      const tr = document.createElement('tr');

      let gateBadge;
      if (r.gate_passed === true)       gateBadge = badge('PASS',  'badge-green');
      else if (r.gate_passed === false)  gateBadge = badge('FAIL',  'badge-red');
      else                               gateBadge = badge('N/A',   'badge-yellow');

      const tdId   = document.createElement('td');
      const tdObs  = document.createElement('td');
      const tdGate = document.createElement('td');
      const tdNote = document.createElement('td');

      tdId.textContent   = r.agent_id;         // textContent — XSS safe
      tdObs.textContent  = r.n_obs;
      tdGate.innerHTML   = gateBadge;           // badge() uses esc() internally
      tdNote.textContent = r.fit_error || '';

      tr.append(tdId, tdObs, tdGate, tdNote);
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    wrap.appendChild(tbl);
  }

  // ── Breakdown ─────────────────────────────────────────────────────
  function renderBreakdown(motifMap, condMap) {
    const wrap = el('breakdown-content');
    wrap.innerHTML = '';
    buildBarGroup(wrap, motifMap, 'By Topology (Motif)');
    buildBarGroup(wrap, condMap,  'By Sharing Condition');
  }

  function buildBarGroup(container, map, label) {
    const div = document.createElement('div');
    div.className = 'breakdown-group';

    const lbl = document.createElement('div');
    lbl.className = 'breakdown-label';
    lbl.textContent = label;
    div.appendChild(lbl);

    Object.entries(map).forEach(([key, { n, passed }]) => {
      const pct = n > 0 ? (passed / n * 100).toFixed(1) : '0.0';

      const row = document.createElement('div');
      row.className = 'bar-row';

      const keyEl = document.createElement('div');
      keyEl.className = 'bar-key';
      keyEl.textContent = key;             // textContent — XSS safe

      const bgEl = document.createElement('div');
      bgEl.className = 'bar-bg';

      const fillEl = document.createElement('div');
      fillEl.className = 'bar-fill';
      fillEl.style.width = pct + '%';
      bgEl.appendChild(fillEl);

      const pctEl = document.createElement('div');
      pctEl.className = 'bar-pct';
      pctEl.textContent = pct + '%';

      row.append(keyEl, bgEl, pctEl);
      div.appendChild(row);
    });

    container.appendChild(div);
  }

  // ── Main ──────────────────────────────────────────────────────────
  function init() {
    const data = window.DASHBOARD_DATA;

    if (!data) {
      el('no-data-msg').classList.remove('hidden');
      return;
    }

    el('main-content').classList.remove('hidden');

    // Meta label in header
    const ts = data.meta.generated_at
      ? new Date(data.meta.generated_at).toLocaleString()
      : '';
    el('meta-label').textContent =
      data.meta.n_missions + ' missions' + (ts ? ' · ' + ts : '');

    renderKPIs(data);
    renderCertification(data.certification);
    renderComposition(data.composition);
    renderDependence(data.dependence);
    renderDrift(data.drift);
    renderBreakdown(data.motif_breakdown, data.condition_breakdown);
  }

  // Run after DOM + scripts loaded; redraw chart on resize
  document.addEventListener('DOMContentLoaded', () => {
    init();
    window.addEventListener('resize', () => {
      if (!window.DASHBOARD_DATA) return;
      const canvas = el('eprocess-canvas');
      if (canvas && window.Charts) {
        Charts.drawEProcessChart(canvas, window.DASHBOARD_DATA.certification);
      }
    });
  });

})();
