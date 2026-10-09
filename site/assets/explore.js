import {
  h, el, lineChart, sparkline, initTheme, fmtPct, fmtNum,
  rateColor, rateTint, showTip, hideTip, isDark, clamp,
} from './viz.js?v=29';
import { loadSource, canLoadSource, peek } from './hfsource.js?v=29';

initTheme();
const $ = (id) => document.getElementById(id);
const CSS = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const PAGE = 60;                       // rows rendered per "show more" step
const state = {
  meta: null, rows: [], filtered: [],
  env: new Set(), model: new Set(), out: new Set(),
  frLo: 0, frHi: 1, jmLo: 0, npLo: 0,
  q: '', sort: 'final', shown: PAGE, sel: null,
};
const curveCache = new Map();          // "env__model" -> [{path, pts}]
const detailCache = new Map();         // "env__model" -> {path: detail}
const redraws = [];
document.addEventListener('themechange', () => redraws.forEach(f => f()));

/* Average graded continuations per probed boundary. A rate estimated from a
   handful of samples snaps to 0 or 1, which manufactures a large apparent jump,
   so this doubles as a reliability weight. */
const gradedPerProbe = (r) => r.nv / Math.max(1, r.np);

const envColor = (id) => {
  const e = state.meta.envs.find(x => x.id === id);
  return e ? (isDark() ? e.color.dark : e.color.light) : CSS('--series-1');
};
const modelColor = (id) => {
  const m = state.meta.models.find(x => x.id === id);
  return m ? (isDark() ? m.color.dark : m.color.light) : CSS('--series-1');
};
const envLabel = (id) => (state.meta.envs.find(x => x.id === id) || {}).label || id;
const modelLabel = (id) => (state.meta.models.find(x => x.id === id) || {}).label || id;

/* ============================================================= bootstrap */
async function main() {
  try {
    const [meta, index] = await Promise.all([
      fetch('data/meta.json').then(r => r.json()),
      fetch('data/index.json').then(r => r.json()),
    ]);
    state.meta = meta;
    state.rows = index.rows;
  } catch (e) {
    $('tlist').appendChild(h('div', { class: 'empty' },
      'Could not load data/. Run build_data.py, then serve this folder over HTTP ' +
      '(for example: python3 -m http.server).'));
    return;
  }

  readUrl();        // restore state first, so the controls below render checked
  buildFilters();
  wire();
  apply();
}

/* ============================================================== filters */
function buildFilters() {
  const countBy = (field, val) => state.rows.filter(r => r[field] === val).length;

  $('f-env').append(...state.meta.envs.map(e => optRow({
    label: e.label, color: envColor(e.id), count: countBy('env', e.id),
    checked: () => state.env.has(e.id),
    toggle: (on) => { on ? state.env.add(e.id) : state.env.delete(e.id); },
  })));

  $('f-model').append(...state.meta.models.map(m => optRow({
    label: m.label, color: modelColor(m.id), count: countBy('model', m.id),
    checked: () => state.model.has(m.id),
    toggle: (on) => { on ? state.model.add(m.id) : state.model.delete(m.id); },
  })));

  const nDec = state.rows.filter(r => r.r1 >= 0.5).length;
  $('f-out').append(
    optRow({
      label: 'Ends deceptive', color: CSS('--deceptive'), count: nDec,
      checked: () => state.out.has('dec'),
      toggle: (on) => { on ? state.out.add('dec') : state.out.delete('dec'); },
    }),
    optRow({
      label: 'Ends honest', color: CSS('--honest'), count: state.rows.length - nDec,
      checked: () => state.out.has('hon'),
      toggle: (on) => { on ? state.out.add('hon') : state.out.delete('hon'); },
    }));
}

function optRow({ label, color, count, checked, toggle }) {
  const box = h('input', { type: 'checkbox' });
  box.checked = checked();
  box.addEventListener('change', () => { toggle(box.checked); state.shown = PAGE; apply(); });
  const lab = h('label', { class: 'opt' }, [
    box,
    color ? h('span', { class: 'sw', style: `background:${color}` }) : null,
    h('span', {}, label),
    h('span', { class: 'ct' }, fmtNum(count)),
  ]);
  return lab;
}

function wire() {
  const rng = (id, key, fmt = (v) => (v / 100).toFixed(2), scale = 100) => {
    const inp = $(id), out = $(id + '-v');
    inp.value = Math.round(state[key] * scale);
    out.textContent = fmt(inp.value);
    inp.addEventListener('input', () => {
      state[key] = +inp.value / scale;
      out.textContent = fmt(inp.value);
      state.shown = PAGE;
      apply();
    });
  };
  rng('fr-lo', 'frLo'); rng('fr-hi', 'frHi');
  rng('jm-lo', 'jmLo');
  rng('np-lo', 'npLo', (v) => String(v), 1);

  let t;
  $('q').addEventListener('input', (e) => {
    clearTimeout(t);
    t = setTimeout(() => { state.q = e.target.value.toLowerCase().trim(); state.shown = PAGE; apply(); }, 130);
  });
  $('sort').addEventListener('change', (e) => { state.sort = e.target.value; apply(); });
  $('reset').addEventListener('click', () => {
    state.env.clear(); state.model.clear(); state.out.clear();
    state.frLo = 0; state.frHi = 1; state.jmLo = 0;
    state.npLo = 0; state.q = ''; state.shown = PAGE;
    $('q').value = '';
    for (const [id, v] of [['fr-lo', 0], ['fr-hi', 100], ['jm-lo', 0], ['np-lo', 0]]) {
      $(id).value = v;
      $(id + '-v').textContent = id === 'np-lo' ? '0' : (v / 100).toFixed(2);
    }
    document.querySelectorAll('.sidebar input[type=checkbox]').forEach(c => { c.checked = false; });
    apply();
  });

  $('d-close').addEventListener('click', closeDetail);
  $('mask').addEventListener('click', closeDetail);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDetail();
    if (!state.sel) return;
    if (e.key === 'j' || e.key === 'ArrowDown') step(1);
    if (e.key === 'k' || e.key === 'ArrowUp') step(-1);
  });

  $('copy-link').addEventListener('click', async () => {
    writeUrl();
    try {
      await navigator.clipboard.writeText(location.href);
      $('copy-link').textContent = 'Copied';
      setTimeout(() => { $('copy-link').textContent = 'Copy view link'; }, 1400);
    } catch (err) {
      $('copy-link').textContent = 'Copy failed';
      setTimeout(() => { $('copy-link').textContent = 'Copy view link'; }, 1400);
    }
  });
}

function step(d) {
  const i = state.filtered.findIndex(r => r.path === state.sel.path);
  const n = state.filtered[i + d];
  if (n) openDetail(n);
}

/* ================================================================ filter */
function apply() {
  const s = state;
  s.filtered = s.rows.filter(r => {
    if (s.env.size && !s.env.has(r.env)) return false;
    if (s.model.size && !s.model.has(r.model)) return false;
    if (s.out.size) {
      const dec = r.r1 >= 0.5;
      if (!((dec && s.out.has('dec')) || (!dec && s.out.has('hon')))) return false;
    }
    if (r.r1 < s.frLo || r.r1 > s.frHi) return false;
    if (s.jmLo > 0 && !(r.jump >= s.jmLo)) return false;
    if (r.np < s.npLo) return false;
    if (s.q) {
      const hay = `${r.id} ${r.env} ${r.model} ${envLabel(r.env)} ${modelLabel(r.model)}`.toLowerCase();
      if (!hay.includes(s.q)) return false;
    }
    return true;
  });

  const cmp = {
    final: (a, b) => b.r1 - a.r1,
    finala: (a, b) => a.r1 - b.r1,
    id: (a, b) => a.id.localeCompare(b.id),
  }[s.sort] || ((a, b) => b.r1 - a.r1);
  s.filtered.sort(cmp);

  $('n-shown').textContent = fmtNum(s.filtered.length);
  $('n-total').textContent = fmtNum(s.rows.length);
  drawList();
  writeUrl();
}

/* ================================================================== list */
function drawList() {
  const box = $('tlist');
  box.innerHTML = '';
  const f = state.filtered;
  if (!f.length) {
    box.appendChild(h('div', { class: 'empty' },
      'No traces match these filters. Try widening a range or clearing a facet.'));
    $('more-wrap').innerHTML = '';
    return;
  }

  const slice = f.slice(0, state.shown);
  // sparklines need the per-cell curve shards; fetch the ones this page needs
  const need = [...new Set(slice.map(r => `${r.env}__${r.model}`))]
    .filter(k => !curveCache.has(k));

  for (const r of slice) box.appendChild(rowEl(r));

  if (need.length) {
    Promise.all(need.map(loadCurves)).then(() => {
      // fill in sparklines for rows that are still on screen
      for (const r of slice) {
        const node = box.querySelector(`[data-path="${cssEscape(r.path)}"] .spark-slot`);
        if (!node || node.dataset.done) continue;
        const pts = getPts(r);
        if (!pts) continue;
        node.innerHTML = '';
        node.appendChild(sparkline(pts));
        node.dataset.done = '1';
      }
    });
  }

  $('more-wrap').innerHTML = '';
  if (f.length > state.shown) {
    $('more-wrap').appendChild(h('button', {
      class: 'ghost-btn', style: 'width:100%;justify-content:center;padding:9px',
      onclick: () => { state.shown += PAGE * 2; drawList(); },
    }, `Show more — ${fmtNum(f.length - state.shown)} remaining`));
  }
}

function cssEscape(s) { return String(s).replace(/["\\]/g, '\\$&'); }

function rowEl(r) {
  const juncPill = r.j != null
    ? h('span', { class: 'pill ' + (r.r1 >= 0.5 ? 'dec' : 'hon') },
        `juncture @ ${r.j}${r.jpos != null ? ` · ${(100 * r.jpos).toFixed(0)}%` : ''}`)
    : h('span', { class: 'pill none' }, 'no juncture');

  const node = h('div', { class: 'trow' + (state.sel && state.sel.path === r.path ? ' sel' : '') }, [
    h('div', { style: 'min-width:0' }, [
      h('div', { class: 'tid' }, r.id),
      h('div', { class: 'tmeta' }, [
        h('span', { class: 'sw', style: `background:${envColor(r.env)}` }),
        h('span', {}, envLabel(r.env)),
        h('span', { style: 'opacity:.5' }, '·'),
        h('span', { class: 'sw', style: `background:${modelColor(r.model)}` }),
        h('span', {}, modelLabel(r.model)),
        h('span', { style: 'opacity:.5' }, '·'),
        h('span', {}, `${r.np} boundaries`),
        h('span', { style: 'opacity:.5' }, '\u00b7'),
        h('span', {
          style: gradedPerProbe(r) < 10 ? 'color:var(--critical);font-weight:600' : '',
          title: 'average graded continuations per probed boundary',
        }, `${Math.round(gradedPerProbe(r))} graded/probe`),
        juncPill,
      ]),
    ]),
    h('div', { class: 'spark-slot', style: 'width:132px;height:30px' }, sparkSlot(r)),
    h('div', { class: 'tnums' }, [
      // the number stays in ink; the swatch beside it carries the rate colour
      h('span', { class: 'big' }, [
        h('span', {
          class: 'dot',
          style: `background:${rateColor(r.r1)}`,
        }),
        document.createTextNode(fmtPct(r.r1)),
      ]),
      h('span', {}, `${arrow(r.r1 - r.r0)} from ${fmtPct(r.r0)}`),
    ]),
  ]);
  node.dataset.path = r.path;
  node.style.setProperty('--spine', rateColor(r.r1));
  node.addEventListener('click', () => openDetail(r));
  return node;
}

/** Direction of travel between the first and last probed boundary. */
function arrow(delta) {
  if (delta > 0.005) return '\u2191';   // rose
  if (delta < -0.005) return '\u2193';  // fell
  return '\u2192';                      // unchanged
}

function sparkSlot(r) {
  const pts = getPts(r);
  return pts ? sparkline(pts) : h('span', {});
}

function getPts(r) {
  const shard = curveCache.get(`${r.env}__${r.model}`);
  if (!shard) return null;
  const rec = shard.byPath.get(r.path);
  return rec ? rec.pts : null;
}

async function loadCurves(key) {
  if (curveCache.has(key)) return curveCache.get(key);
  const items = await fetch(`data/curves/${key}.json`).then(r => r.json());
  const obj = { items, byPath: new Map(items.map(i => [i.path, i])) };
  curveCache.set(key, obj);
  return obj;
}

async function loadDetail(key) {
  if (detailCache.has(key)) return detailCache.get(key);
  let obj = null;
  try {
    const r = await fetch(`data/detail/${key}.json`);
    obj = r.ok ? await r.json() : {};
  } catch (e) { obj = {}; }
  detailCache.set(key, obj);
  return obj;
}

/* ================================================================ detail */
async function openDetail(r) {
  state.sel = r;
  document.querySelectorAll('.trow').forEach(n =>
    n.classList.toggle('sel', n.dataset.path === r.path));

  $('detail').classList.add('on');
  $('detail').setAttribute('aria-hidden', 'false');
  $('mask').classList.add('on');
  $('d-title').textContent = `${envLabel(r.env)} · ${modelLabel(r.model)}`;
  $('d-id').textContent = r.id;
  const body = $('d-body');
  body.innerHTML = '';
  body.appendChild(h('div', { class: 'empty' }, [h('span', { class: 'spinner' }), ' Loading trace…']));
  writeUrl();

  const key = `${r.env}__${r.model}`;
  const [curves, detail] = await Promise.all([loadCurves(key), loadDetail(key)]);
  if (state.sel !== r) return;                       // user moved on
  const rec = curves.byPath.get(r.path);
  const det = detail && detail.t ? detail.t[r.path] : null;
  if (det) det._prompt = detail.p[det.pi] || '';
  // Draw straight away from the prebuilt index so the drawer is never blank,
  // then read the trace's own file from the Hub and redraw from that. Everything
  // shown here - curve, counts, continuations - then comes from the source file.
  renderDetail(r, rec, det, peek(r.path));
  if (!peek(r.path) && canLoadSource()) {
    loadSource(r.path)
      .then((probes) => { if (state.sel === r) renderDetail(r, rec, det, probes); })
      .catch((err) => console.warn('[explore] live source unavailable:', err.message));
  }
}

function closeDetail() {
  state.sel = null;
  $('detail').classList.remove('on');
  $('detail').setAttribute('aria-hidden', 'true');
  $('mask').classList.remove('on');
  document.querySelectorAll('.trow').forEach(n => n.classList.remove('sel'));
  writeUrl();
}

function renderDetail(r, rec, det, live) {
  const body = $('d-body');
  body.innerHTML = '';
  if (!rec && !live) {
    body.appendChild(h('div', { class: 'empty' }, 'Curve data unavailable.'));
    return;
  }
  // Prefer the source file once it has arrived; the index is only a stand-in
  // until then. Both carry the same fields, so nothing downstream changes.
  const pts = live
    ? [...live.values()]
        .filter(p => p.rate != null && p.numValid > 0)
        .sort((a, b) => a.i - b.i)
        .map(p => ({ i: p.i, r: p.rate, lo: p.ciLow, hi: p.ciHigh,
                     nv: p.numValid, nt: p.numTruthful, s: p.sentence }))
    : rec.pts;
  if (!pts.length) {
    body.appendChild(h('div', { class: 'empty' }, 'No gradeable boundaries.'));
    return;
  }
  const gradedTotal = pts.reduce((a, p) => a + p.nv, 0);
  const stat = {
    r0: pts[0].r,
    r1: pts[pts.length - 1].r,
    gpp: gradedTotal / Math.max(1, pts.length),
    jump: pts.slice(1).reduce((best, p, i) =>
      Math.abs(p.r - pts[i].r) > Math.abs(best) ? p.r - pts[i].r : best, 0),
  };

  /* --- key numbers -------------------------------------------------- */
  body.appendChild(h('div', { class: 'stats six', style: 'margin-top:0' }, [
    ['first boundary', fmtPct(stat.r0)],
    ['final boundary', fmtPct(stat.r1)],
    ['juncture', r.j != null ? `sentence ${r.j}` : 'none'],
    ['position', r.jpos != null ? fmtPct(r.jpos) : '—'],
    ['sharpest jump', (stat.jump >= 0 ? '+' : '') + (100 * stat.jump).toFixed(0) + ' pts'],
    ['graded / probe', String(Math.round(stat.gpp))],
  ].map(([l, v]) => h('div', { class: 'stat' }, [
    h('div', { class: 'v', style: 'font-size:18px' }, v),
    h('div', { class: 'l' }, l),
  ]))));

  if (stat.gpp < 10) {
    body.appendChild(h('div', {
      style: 'font-size:12.5px;border:1px solid var(--critical);border-radius:var(--radius);'
           + 'padding:10px 12px;color:var(--text-secondary)',
    }, [
      h('b', { style: 'color:var(--critical)' }, 'Thinly sampled. '),
      document.createTextNode(
        `Only about ${Math.round(stat.gpp)} continuations per boundary could be graded here, `
        + 'against ~50 sampled. Rates estimated from so few land on 0 or 1 easily, so this trace\u2019s '
        + 'curve and its jump are far less reliable than a typical one. About 2% of traces are like this.'),
    ]));
  }

  /* --- the curve ---------------------------------------------------- */
  const curveBox = h('div', { class: 'figure' });
  curveBox.appendChild(h('div', { class: 'fig-head' }, [
    h('h3', {}, 'Deception rate at each probed boundary'),
    h('p', {}, 'Shaded band is the Wilson interval. Hover for the sentence and its counts.'),
  ]));
  const plot = h('div');
  curveBox.appendChild(plot);
  body.appendChild(curveBox);

  const drawCurve = () => lineChart(plot, [{
    key: 'r', label: '', color: CSS('--text-primary'),
    pts: pts.map(p => ({ x: p.i, y: p.r, lo: p.lo, hi: p.hi, n: p.nv, nt: p.nt, s: p.s })),
  }], {
    width: 840, height: 230, yDomain: [0, 1],
    xLabel: 'sentence boundary', yLabel: 'deception rate',
    xFmt: (v) => String(Math.round(v)), rule: 0.5, ruleLabel: '0.5',
    vrule: r.j != null ? r.j + 1 : null, vruleLabel: r.j != null ? 'juncture' : '',
    directLabel: false, margin: { t: 20, r: 34, b: 38, l: 48 },
    onPick: (p) => {
      const panel = body.querySelector('[data-continuations]');
      if (panel && panel.__pick) panel.__pick(Math.round(p.x));
    },
    tipFmt: (p) => `<div class="t-head"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${rateColor(p.y)}"></span>Boundary ${Math.round(p.x)}</div>` +
      `<div class="t-row"><span>deception rate</span><b>${fmtPct(p.y, 1)}</b></div>` +
      `<div class="t-row"><span>95% CI</span><b>${fmtPct(p.lo, 1)} &ndash; ${fmtPct(p.hi, 1)}</b></div>` +
      `<div class="t-row"><span>continuations</span><b>${p.n - p.nt} deceptive / ${p.n} graded</b></div>` +
      `<div class="t-sent">${escapeHtml(p.s).slice(0, 220)}</div>`,
  });
  drawCurve();
  redraws.push(drawCurve);

  /* --- scale key ---------------------------------------------------- */
  const grad = `linear-gradient(90deg, ${[0, .17, .34, .5, .66, .83, 1].map(v => rateColor(v)).join(',')})`;
  body.appendChild(h('div', { class: 'scale-strip' }, [
    h('span', {}, 'all continuations honest'),
    h('span', { class: 'scale-bar', style: `background:${grad}` }),
    h('span', {}, 'all deceptive'),
  ]));

  /* --- the reasoning trace ------------------------------------------ */
  const traceBox = h('div', { class: 'figure' });
  traceBox.appendChild(h('div', { class: 'fig-head' }, [
    h('h3', {}, 'The reasoning trace'),
    h('p', {}, det && det.raw
      ? 'The complete trace, tinted by the rate measured at the boundary each sentence ends. '
        + 'Hover for that rate; click a sentence to read every continuation sampled from it. '
        + 'Grey text was never probed.'
      : 'Probed sentences only, in order. Hover for the rate at each boundary; click one to read '
        + 'its continuations.'),
  ]));
  traceBox.appendChild(traceText(pts, det, r, live));
  body.appendChild(traceBox);

  /* --- continuations browser --------------------------------------- */
  body.appendChild(continuationsPanel(r, pts, det, live));

  /* --- prompt & provenance ------------------------------------------ */
  const prov = h('details', { class: 'table-toggle' }, [
    h('summary', {}, 'Prompt and source file'),
  ]);
  if (det && det._prompt) {
    prov.appendChild(h('div', {
      class: 'g-text', style: 'max-height:260px;border:1px solid var(--border);' +
        'border-radius:var(--radius-sm);padding:10px;background:var(--surface-0)',
    }, det._prompt));
  }
  if (det && det.ctx) {
    prov.appendChild(h('div', { style: 'margin-top:10px' }, [
      h('div', { style: 'font-size:11.5px;color:var(--text-muted);margin-bottom:4px' }, 'eval_context'),
      h('div', { class: 'mono', style: 'font-size:11.5px;color:var(--text-secondary)' },
        JSON.stringify(det.ctx)),
    ]));
  }
  prov.appendChild(h('div', { style: 'margin-top:10px' }, [
    h('div', { style: 'font-size:11.5px;color:var(--text-muted);margin-bottom:4px' }, 'source file'),
    h('a', {
      class: 'mono', style: 'font-size:11.5px;word-break:break-all',
      href: `${state.meta.repo_url}/blob/main/${r.path}`,
      target: '_blank', rel: 'noopener',
    }, r.path),
  ]));
  body.appendChild(prov);

  body.appendChild(h('div', { style: 'font-size:12px;color:var(--text-muted);padding-top:4px' },
    [document.createTextNode('Press '), h('kbd', {}, 'J'), document.createTextNode(' / '),
     h('kbd', {}, 'K'), document.createTextNode(' to move through the filtered list, '),
     h('kbd', {}, 'Esc'), document.createTextNode(' to close.')]));
}

/* ===========================================================================
   Continuations browser
   ---------------------------------------------------------------------------
   Shows, for one probed boundary, the continuations sampled from that frozen
   prefix and how each was graded. The site bundles a handful around the
   juncture for an instant view; the full set (~50 per boundary, every boundary)
   is fetched from the Hub on request.
   =========================================================================== */
function continuationsPanel(r, pts, det, live) {
  const box = h('div', { class: 'figure' });
  const bundled = new Map(
    ((det && det.pr) || []).filter(p => p.g && p.g.length).map(p => [p.i, p.g]));

  // start on the juncture's boundary when there is one
  let sel = (r.j != null && pts.some(p => p.i === r.j + 1)) ? r.j + 1
          : (bundled.size ? [...bundled.keys()][0] : (pts[0] || {}).i);
  let full = live || peek(r.path);
  let autoFailed = null;
  let autoNote = null;
  let focus = null;   // continuation number pinned open from the grid
  const show = { deceptive: true, honest: true, ungraded: true };

  const head = h('div', { class: 'fig-head' }, [
    h('h3', {}, 'Continuations from a frozen prefix'),
    h('p', {}, 'Every continuation was sampled independently from the same prefix, then graded '
      + 'by the environment rule. The deception rate is the share of deceptive among those that '
      + 'could be graded.'),
  ]);
  const chipRow = h('div', { style: 'display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin-bottom:12px' });
  const loadRow = h('div', { style: 'margin-bottom:12px' });
  const gridRow = h('div');
  const bodyRow = h('div');
  box.append(head, chipRow, loadRow, gridRow, bodyRow);
  box.setAttribute('data-continuations', '');
  // let the curve above drive the boundary selection
  box.__pick = (i) => {
    if (!pts.some(p => p.i === i)) return;
    sel = i; focus = null;
    drawChips(); drawGrid(); drawBody();
    box.scrollIntoView({ block: 'nearest' });
  };

  /* ---- boundary chips ------------------------------------------------ */
  function drawChips() {
    chipRow.innerHTML = '';
    chipRow.appendChild(h('span', {
      style: 'font-size:11.5px;color:var(--text-muted);align-self:center;margin-right:3px',
    }, 'boundary'));
    for (const p of pts) {
      const has = full ? full.has(p.i) : bundled.has(p.i);
      const isJ = r.j === p.i - 1;
      const on = p.i === sel;
      const chip = h('button', {
        class: 'chip' + (on ? ' on' : '') + (has ? '' : ' chip-muted'),
        title: `Boundary ${p.i} — ${fmtPct(p.r, 1)} deceptive`
             + (isJ ? ' (commitment juncture)' : '')
             + (has ? '' : ' — load from Hugging Face to see these'),
        onclick: () => { sel = p.i; focus = null; drawChips(); drawGrid(); drawBody(); },
      }, [
        h('span', { class: 'chip-dot', style: `background:${rateColor(p.r)}` }),
        document.createTextNode(String(p.i)),
        isJ ? h('span', { class: 'chip-star', title: 'commitment juncture' }, '★') : null,
      ]);
      chipRow.appendChild(chip);
    }
  }

  /* ---- load-from-Hub control ----------------------------------------- */
  function drawLoad() {
    loadRow.innerHTML = '';
    if (full) {
      const n = [...full.values()].reduce((a, p) => a + p.gens.length, 0);
      loadRow.appendChild(h('div', { style: 'font-size:12.5px;color:var(--text-secondary)' }, [
        h('span', { class: 'pill hon', title: 'The curve, the counts and these continuations '
          + 'are all read from this trace\u2019s own file on the Hub.' }, 'live from Hugging Face'),
        document.createTextNode(` ${fmtNum(n)} continuations across ${full.size} boundaries, `),
        h('a', { href: sourceUrl(r.path), target: '_blank', rel: 'noopener' }, 'from the Hub'),
        document.createTextNode('.'),
      ]));
      return;
    }
    if (!canLoadSource()) {
      loadRow.appendChild(h('div', { style: 'font-size:12.5px;color:var(--text-muted)' }, [
        document.createTextNode('This browser cannot decompress the source file in-page, so only the '
          + 'bundled sample is shown. Every continuation is in the '),
        h('a', { href: sourceUrl(r.path), target: '_blank', rel: 'noopener' }, 'source file'),
        document.createTextNode('.'),
      ]));
      return;
    }
    if (!autoFailed) {
      // fetch is already in flight - say so rather than showing a button
      autoNote = h('div', { style: 'font-size:12.5px;color:var(--text-secondary)' }, [
        h('span', { class: 'spinner' }),
        document.createTextNode(' Loading all continuations\u2026'),
      ]);
      loadRow.appendChild(autoNote);
      return;
    }
    const btn = h('button', { class: 'ghost-btn' },
      'Retry loading every continuation from Hugging Face');
    const note = h('span', {
      style: 'font-size:12px;color:var(--critical);margin-left:9px',
    }, `Automatic load failed: ${autoFailed}. Showing the bundled sample.`);
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.innerHTML = '';
      btn.append(h('span', { class: 'spinner' }), document.createTextNode(' Loading…'));
      try {
        full = await loadSource(r.path, (loaded, total) => {
          const pct = total ? ` ${Math.round(100 * loaded / total)}%` : '';
          btn.lastChild.textContent = ` Loading${pct}…`;
        });
        drawChips(); drawLoad(); drawGrid(); drawBody();
      } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Retry loading every continuation from Hugging Face';
        note.textContent = `Could not load: ${e.message}`;
      }
    });
    loadRow.append(btn, note);
  }

  /* ---- every continuation as one tile --------------------------------- */
  function drawGrid() {
    gridRow.innerHTML = '';
    const gens = gensFor(sel);
    if (!gens.length) return;

    const wrap = h('div', { style: 'margin-bottom:12px' });
    wrap.appendChild(h('div', {
      style: 'font-size:11.5px;color:var(--text-muted);margin-bottom:6px',
    }, `${gens.length} continuations sampled here \u2014 one tile each, `
     + 'in the order they were drawn. Click any tile to read it.'));

    const cells = h('div', { class: 'cellgrid' });
    for (const g of gens) {
      const cls = g.label === 'deceptive' ? 'd' : g.label === 'honest' ? 'h' : 'u';
      const cell = h('button', {
        class: `cell ${cls}` + (focus === g.n ? ' on' : ''),
        'aria-label': `continuation ${g.n}, ${g.label}`,
        onclick: () => { focus = focus === g.n ? null : g.n; drawGrid(); drawBody(); },
      });
      cell.addEventListener('mousemove', (ev) => showTip(
        `<div class="t-head"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;`
        + `background:${g.label === 'deceptive' ? 'var(--deceptive)'
                        : g.label === 'honest' ? 'var(--honest)' : 'var(--border-strong)'}"></span>`
        + `#${g.n} \u00b7 ${g.label === 'ungraded' ? 'could not be graded' : g.label}</div>`
        + `<div class="t-sent">${escapeHtml((g.text || '').slice(0, 200))}\u2026</div>`
        + `<div class="t-row" style="margin-top:5px;opacity:.75"><span>click to open</span></div>`, ev));
      cell.addEventListener('mouseleave', hideTip);
      cells.appendChild(cell);
    }
    wrap.appendChild(cells);
    gridRow.appendChild(wrap);
  }

  /** Continuations for a boundary: the full set once loaded, else the bundle. */
  function gensFor(i) {
    const src = full ? full.get(i) : null;
    const RANK = { deceptive: 0, honest: 1, ungraded: 2 };
    if (src) return src.gens;
    return (bundled.get(i) || []).map((g, k) => ({
      n: k + 1, text: g.t, label: g.d ? 'deceptive' : 'honest',
      lossy: !!g.lossy, evaluation: g.e || null, err: null,
    }));
  }

  /* ---- the selected boundary ----------------------------------------- */
  function drawBody() {
    bodyRow.innerHTML = '';
    const probe = pts.find(p => p.i === sel);
    const src = full ? full.get(sel) : null;
    const RANK = { deceptive: 0, honest: 1, ungraded: 2 };
    const gens = gensFor(sel).slice()
      .sort((a, b) => RANK[a.label] - RANK[b.label] || a.n - b.n);

    // the prefix this boundary froze
    if (probe) bodyRow.appendChild(prefixLine(probe, full, det));

    const nDec = gens.filter(g => g.label === 'deceptive').length;
    const nHon = gens.filter(g => g.label === 'honest').length;
    const nUng = gens.filter(g => g.label === 'ungraded').length;
    const graded = nDec + nHon;

    // stacked share bar over the graded continuations
    if (gens.length) {
      const seg = (n, colour, title) => n ? h('span', {
        title, style: `flex:${n} 0 0;background:${colour};height:100%`,
      }) : null;
      bodyRow.appendChild(h('div', { style: 'margin-bottom:10px' }, [
        h('div', {
          style: 'display:flex;height:10px;border-radius:3px;overflow:hidden;gap:2px;'
               + 'background:var(--surface-3);margin-bottom:6px',
        }, [
          seg(nDec, 'var(--deceptive)', `${nDec} deceptive`),
          seg(nHon, 'var(--honest)', `${nHon} honest`),
          seg(nUng, 'var(--border-strong)', `${nUng} could not be graded`),
        ]),
        h('div', { style: 'font-size:12.5px;color:var(--text-secondary)' },
          graded
            ? `${fmtPct(nDec / graded, 1)} deceptive — ${nDec} of ${graded} graded`
              + (nUng ? `, ${nUng} of ${gens.length} could not be graded` : '')
            : `none of these ${gens.length} could be graded`),
      ]));
    }

    const nRec = gens.filter(g => g.recoverable).length;
    if (nRec) {
      bodyRow.appendChild(h('div', {
        style: 'font-size:12px;color:var(--text-secondary);border:1px solid var(--warning);'
             + 'border-radius:var(--radius-sm);padding:8px 10px;margin-bottom:10px',
      }, [
        h('b', {}, `${nRec} of the ${nUng} ungraded here are valid JSON. `),
        document.createTextNode(
          'The grader parsed them before decoding byte-level BPE symbols, so it rejected '
          + 'well-formed output. They are excluded from the rate above. Seen only in '
          + 'DeepSeek-R1-Distill-Llama-8B.'),
      ]));
    }

    // filter chips
    const counts = { deceptive: nDec, honest: nHon, ungraded: nUng };
    const filters = h('div', { style: 'display:flex;gap:6px;flex-wrap:wrap;margin-bottom:11px' });
    for (const k of ['deceptive', 'honest', 'ungraded']) {
      if (!counts[k]) continue;
      filters.appendChild(h('button', {
        class: 'chip' + (show[k] ? ' on' : ''),
        onclick: () => { show[k] = !show[k]; drawBody(); },
      }, [
        h('span', {
          class: 'chip-dot',
          style: `background:${k === 'deceptive' ? 'var(--deceptive)'
                              : k === 'honest' ? 'var(--honest)' : 'var(--border-strong)'}`,
        }),
        document.createTextNode(`${k} ${counts[k]}`),
      ]));
    }
    bodyRow.appendChild(filters);

    // a tile clicked in the grid pins that one continuation on its own
    if (focus != null) {
      const g = gens.find(x => x.n === focus);
      if (g) {
        bodyRow.appendChild(h('div', {
          style: 'display:flex;align-items:center;gap:9px;margin-bottom:8px',
        }, [
          h('b', { style: 'font-size:13px' }, `Continuation #${g.n}`),
          h('button', { class: 'chip', onclick: () => { focus = null; drawGrid(); drawBody(); } },
            'show all again'),
        ]));
        bodyRow.appendChild(genCard(g));
        return;
      }
      focus = null;
    }

    const visible = gens.filter(g => show[g.label]);
    if (!visible.length) {
      bodyRow.appendChild(h('div', { class: 'empty', style: 'padding:24px' },
        gens.length ? 'No continuations match these labels.'
                    : 'No continuations bundled for this boundary — load the full source above.'));
      return;
    }

    const list = h('div');
    visible.forEach(g => list.appendChild(genCard(g)));
    bodyRow.appendChild(list);

    if (!full && bundled.has(sel)) {
      bodyRow.appendChild(h('div', {
        style: 'font-size:12px;color:var(--text-muted);margin-top:10px',
      }, `Showing ${gens.length} of about 50 sampled here. Load the full source above for all of them, `
       + 'and for every other boundary.'));
    }
  }


  drawChips(); drawLoad(); drawGrid(); drawBody();

  // The bundled sample exists only so something renders instantly; the full set
  // of continuations is what the reader actually wants, so fetch it straight
  // away rather than making them ask. The bundled view stays put until it lands,
  // and remains the fallback if the fetch fails.
  if (!full && canLoadSource()) {
    autoLoad();
  }

  async function autoLoad() {
    try {
      const probes = await loadSource(r.path, (loaded, total) => {
        if (state.sel !== r) return;
        const pct = total ? ` ${Math.round(100 * loaded / total)}%` : '';
        if (autoNote) autoNote.lastChild.textContent = ` Loading all continuations\u2026${pct}`;
      });
      if (state.sel !== r) return;         // reader moved on
      full = probes;
      drawChips(); drawLoad(); drawGrid(); drawBody();
    } catch (e) {
      if (state.sel !== r) return;
      autoFailed = e.message || 'fetch failed';
      drawLoad();
    }
  }

  return box;
}

function sourceUrl(path) {
  return `https://huggingface.co/datasets/a91939448/deception-localization/blob/main/${path}`;
}

/* Everything sampled from one frozen prefix, rendered to sit inline under the
   sentence that was clicked. The separate panel further down does the same job
   for readers who scroll; this exists so a click has an answer on the spot. */
function inlineGenerations(probe, gens, onClose, live, det) {
  const box = h('div', { class: 'inline-gens' });
  const nDec = gens.filter(g => g.label === 'deceptive').length;
  const nHon = gens.filter(g => g.label === 'honest').length;
  const nUng = gens.filter(g => g.label === 'ungraded').length;
  const graded = nDec + nHon;

  box.appendChild(h('div', { class: 'ig-head' }, [
    h('b', {}, `Boundary ${probe.i}`),
    h('span', { style: 'color:var(--text-secondary)' },
      graded ? `${fmtPct(nDec / graded, 1)} deceptive \u00b7 ${nDec} of ${graded} graded`
             : `none of these ${gens.length} could be graded`),
    nUng ? h('span', { style: 'color:var(--text-muted)' }, `\u00b7 ${nUng} ungradeable`) : null,
    h('button', { class: 'chip', style: 'margin-left:auto', onclick: onClose }, 'close'),
  ]));

  box.appendChild(prefixLine(probe, live, det));

  const seg = (n, colour, title) => n ? h('span', {
    title, style: `flex:${n} 0 0;background:${colour};height:100%`,
  }) : null;
  box.appendChild(h('div', {
    style: 'display:flex;height:8px;border-radius:3px;overflow:hidden;gap:2px;'
         + 'background:var(--surface-3);margin:8px 0 10px',
  }, [
    seg(nDec, 'var(--deceptive)', `${nDec} deceptive`),
    seg(nHon, 'var(--honest)', `${nHon} honest`),
    seg(nUng, 'var(--border-strong)', `${nUng} ungradeable`),
  ]));

  // one tile per continuation; clicking a tile scrolls its card into view
  const list = h('div', { class: 'ig-list' });
  const cards = new Map();
  const cells = h('div', { class: 'cellgrid', style: 'margin-bottom:10px' });
  const RANK = { deceptive: 0, honest: 1, ungraded: 2 };
  const ordered = gens.slice().sort((a, b) => RANK[a.label] - RANK[b.label] || a.n - b.n);
  for (const g of ordered) {
    const card = genCard(g);
    cards.set(g.n, card);
    list.appendChild(card);
  }
  for (const g of gens) {
    const cls = g.label === 'deceptive' ? 'd' : g.label === 'honest' ? 'h' : 'u';
    const cell = h('button', {
      class: `cell ${cls}`,
      'aria-label': `continuation ${g.n}, ${g.label}`,
      onclick: () => {
        cells.querySelectorAll('.cell.on').forEach(n => n.classList.remove('on'));
        cell.classList.add('on');
        const card = cards.get(g.n);
        list.querySelectorAll('.gen.hit').forEach(n => n.classList.remove('hit'));
        card.classList.add('hit');
        list.scrollTop = card.offsetTop - list.offsetTop;
      },
    });
    cell.addEventListener('mousemove', (ev) => showTip(
      `<div class="t-head">#${g.n} \u00b7 ${g.label === 'ungraded' ? 'could not be graded' : g.label}</div>`
      + `<div class="t-sent">${escapeHtml((g.text || '').slice(0, 200))}\u2026</div>`, ev));
    cell.addEventListener('mouseleave', hideTip);
    cells.appendChild(cell);
  }
  box.appendChild(cells);
  box.appendChild(list);
  return box;
}

/* "Prefix: … <closing sentence>", where the ellipsis opens the whole frozen
   prefix. The prefix is the thing actually held fixed before resampling, so it
   is what the reader needs when asking why the continuations went the way they
   did - the closing sentence alone is only the last step of it. */
function prefixLine(probe, live, det) {
  const rec = live ? live.get(probe.i) : null;
  let full = rec && rec.prefix ? rec.prefix : null;
  if (!full && det && det.raw) {
    // no source file yet: rebuild it from the trace text and this probe's span
    const p = ((det.pr) || []).find(x => x.i === probe.i);
    if (p && p.sp) full = det.raw.slice(0, p.sp[1]);
  }
  const tail = probe.s || '';
  const earlier = full && full.endsWith(tail) ? full.slice(0, full.length - tail.length) : full;
  const expandable = !!(earlier && earlier.trim());

  const box = h('div', { class: 'prefix-line' });
  const body = h('div', { class: 'pl-body' });
  let open = false;

  const render = () => {
    body.innerHTML = '';
    if (open) {
      body.appendChild(h('div', { class: 'pl-full' }, [
        h('span', { class: 'pl-earlier' }, earlier),
        h('span', { class: 'pl-tail' }, tail),
      ]));
      body.appendChild(h('button', {
        class: 'chip', style: 'margin-top:7px',
        onclick: () => { open = false; render(); },
      }, 'show less'));
    } else {
      body.appendChild(h('span', {}, [
        expandable
          ? h('button', {
              class: 'pl-dots',
              title: `Show the whole frozen prefix (${fmtNum(full.length)} characters)`,
              onclick: () => { open = true; render(); },
            }, '\u2026')
          : document.createTextNode('\u2026'),
        document.createTextNode(tail),
      ]));
    }
  };
  render();

  box.append(h('span', { class: 'pl-label' }, 'Prefix'), body);
  return box;
}

function genCard(g) {
  const cls = g.label === 'deceptive' ? 'd' : g.label === 'honest' ? 'h' : 'u';
  const pill = g.label === 'deceptive' ? 'dec' : g.label === 'honest' ? 'hon' : 'none';
  return h('div', { class: `gen ${cls}` }, [
    h('div', { class: 'g-head' }, [
      h('span', { class: `pill ${pill}` }, g.label === 'ungraded' ? 'could not be graded' : g.label),
      h('span', {
        style: 'font-size:10.5px;color:var(--text-muted)',
        title: 'position in the original sample',
      }, `#${g.n}`),
      g.recoverable ? h('span', {
        class: 'pill warn',
        title: 'The grader ran a JSON parse over text that still held byte-level BPE '
             + 'symbols, so it rejected output that is valid JSON once decoded. This '
             + 'continuation was dropped from the rate for that reason, not because the '
             + 'model produced anything malformed.',
      }, 'valid JSON, rejected by the grader') : null,
      g.lossy ? h('span', {
        class: 'pill none',
        title: 'The stored text lost its word-separator tokens upstream. '
             + 'The label and the rate are unaffected.',
      }, 'spacing lost in source') : null,
      g.evaluation ? h('span', {
        class: 'mono g-eval', title: g.evaluation,
      }, g.evaluation) : null,
    ]),
    g.err ? h('div', {
      class: 'mono',
      style: 'font-size:11px;color:var(--text-muted);margin-bottom:4px',
    }, `parse_error: ${g.err}`) : null,
    h('div', { class: 'g-text' }, g.text || '(empty)'),
  ]);
}

/* Render the trace with probed sentences tinted. When the full raw text is
   available we show it whole and tint only the probed spans, so the reader sees
   the sentences that were never probed too. */
function traceText(pts, det, r, live) {
  const box = h('div', { class: 'trace-text' });
  let openAt = null;                       // boundary whose expansion is showing

  const bundledAt = (i) => {
    const p = ((det && det.pr) || []).find(x => x.i === i);
    return (p && p.g ? p.g : []).map((g, k) => ({
      n: k + 1, text: g.t, label: g.d ? 'deceptive' : 'honest',
      lossy: !!g.lossy, evaluation: g.e || null, err: null,
    }));
  };
  const gensAt = (i) => {
    const src = live ? live.get(i) : null;
    return src ? src.gens : bundledAt(i);
  };
  const closeInline = () => {
    box.querySelectorAll('.inline-gens').forEach(n => n.remove());
    box.querySelectorAll('.sent.picked').forEach(n => n.classList.remove('picked'));
    openAt = null;
  };
  const byIdx = new Map(pts.map(p => [p.i, p]));

  const addSent = (text, probe) => {
    if (!text) return;
    const span = h('span', {
      class: 'sent' + (probe ? '' : ' unprobed'),
      style: probe ? `background:${rateTint(probe.r)}` : '',
    }, text + ' ');
    if (probe) {
      span.addEventListener('mousemove', (evt) => showTip(
        `<div class="t-head"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${rateColor(probe.r)}"></span>Boundary ${probe.i}${r.j === probe.i - 1 ? ' · juncture' : ''}</div>` +
        `<div class="t-row"><span>deception rate</span><b>${fmtPct(probe.r, 1)}</b></div>` +
        `<div class="t-row"><span>95% CI</span><b>${fmtPct(probe.lo, 1)} &ndash; ${fmtPct(probe.hi, 1)}</b></div>` +
        `<div class="t-row"><span>continuations</span><b>${probe.nv - probe.nt} deceptive / ${probe.nv} graded</b></div>`,
        evt));
      span.addEventListener('mouseleave', hideTip);
      // the most natural place to ask "what did it say next?" is the sentence
      // itself, so clicking one drives the continuations panel
      span.addEventListener('click', () => {
        const wasOpen = openAt === probe.i;
        closeInline();
        if (wasOpen) return;               // clicking the same sentence closes it
        openAt = probe.i;
        box.querySelectorAll(`[data-boundary="${probe.i}"]`)
           .forEach(n => n.classList.add('picked'));
        const gens = gensAt(probe.i);
        const panel = h('div');
        if (!gens.length) {
          panel.appendChild(h('div', { class: 'inline-gens' }, [
            h('div', { class: 'ig-head' }, [
              h('b', {}, `Boundary ${probe.i}`),
              h('span', { style: 'color:var(--text-secondary)' }, [
                h('span', { class: 'spinner' }),
                document.createTextNode(' loading continuations from Hugging Face\u2026'),
              ]),
            ]),
          ]));
        } else {
          panel.appendChild(inlineGenerations(probe, gens, closeInline, live, det));
        }
        // place it after the last fragment of this sentence, so it reads as an
        // expansion of the sentence rather than an interruption of the next one
        const frags = [...box.querySelectorAll(`[data-boundary="${probe.i}"]`)];
        const last = frags[frags.length - 1];
        last.after(panel.firstChild);
        // keep the panel below in step for readers who scroll
        const side = document.querySelector('[data-continuations]');
        if (side && side.__pick) side.__pick(probe.i);
      });
      span.dataset.boundary = probe.i;
      if (r.j === probe.i - 1) span.classList.add('active');
    }
    box.appendChild(span);
  };

  if (det && det.raw && det.pr && det.pr.length) {
    // Walk the raw trace, tinting probed spans and showing the gaps between
    // them as plain unprobed text, so the reader sees the whole reasoning.
    const spans = det.pr.filter(p => p.sp)
      .map(p => ({ a: p.sp[0], b: p.sp[1], probe: byIdx.get(p.i) }))
      .sort((x, y) => x.a - y.a);
    let cur = 0;
    for (const sp of spans) {
      if (sp.a > cur) addSent(det.raw.slice(cur, sp.a).trim(), null);
      addSent(det.raw.slice(sp.a, sp.b), sp.probe);
      cur = Math.max(cur, sp.b);
    }
    if (cur < det.raw.length) addSent(det.raw.slice(cur).trim(), null);
    // any probe whose span did not resolve still gets shown
    for (const p of det.pr) if (!p.sp && p.s) addSent(p.s, byIdx.get(p.i));
  } else if (det && det.pr && det.pr.length) {
    for (const p of det.pr) addSent(probeText(p, det), byIdx.get(p.i));
  } else {
    for (const p of pts) addSent(p.s, p);
  }
  return box;
}

/** A probe's sentence: stored inline when the span did not resolve, otherwise
    sliced back out of the raw trace. */
function probeText(p, det) {
  if (p.s != null) return p.s;
  if (p.sp && det && det.raw) return det.raw.slice(p.sp[0], p.sp[1]);
  return '';
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* ============================================================ URL state */
function writeUrl() {
  const p = new URLSearchParams();
  if (state.env.size) p.set('env', [...state.env].join(','));
  if (state.model.size) p.set('model', [...state.model].join(','));
  if (state.out.size) p.set('out', [...state.out].join(','));
  if (state.frLo > 0) p.set('frlo', state.frLo.toFixed(2));
  if (state.frHi < 1) p.set('frhi', state.frHi.toFixed(2));
  if (state.jmLo > 0) p.set('jmlo', state.jmLo.toFixed(2));
  if (state.npLo > 0) p.set('nplo', String(state.npLo));
  if (state.q) p.set('q', state.q);
  if (state.sort !== 'final') p.set('sort', state.sort);
  if (state.sel) p.set('trace', state.sel.path);
  const url = location.pathname + (p.toString() ? '?' + p : '');
  history.replaceState(null, '', url);
}

function readUrl() {
  const p = new URLSearchParams(location.search);
  const setOf = (k, target) => {
    const v = p.get(k);
    if (v) v.split(',').filter(Boolean).forEach(x => target.add(x));
  };
  setOf('env', state.env); setOf('model', state.model);
  setOf('out', state.out);
  const num = (k, d) => (p.has(k) ? +p.get(k) : d);
  state.frLo = num('frlo', 0); state.frHi = num('frhi', 1);
  state.jmLo = num('jmlo', 0); state.npLo = num('nplo', 0);
  state.q = p.get('q') || '';
  const SORTS = new Set(['final', 'finala', 'id']);
  const wanted = p.get('sort');
  state.sort = SORTS.has(wanted) ? wanted : 'final';
  $('q').value = state.q;
  $('sort').value = state.sort;

  const pending = p.get('trace');
  if (pending) {
    setTimeout(() => {
      const r = state.rows.find(x => x.path === pending);
      if (r) openDetail(r);
    }, 0);
  }
}

main();
