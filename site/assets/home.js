import {
  h, el, lineChart, initTheme,
  fmtPct, fmtNum, rateColor, rateTint, showTip, hideTip, isDark,
} from './viz.js?v=29';

initTheme();

const $ = (id) => document.getElementById(id);
const redraws = [];
const onRedraw = (fn) => { redraws.push(fn); fn(); };
document.addEventListener('themechange', () => redraws.forEach(f => f()));

const CSS = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* Each section of the page renders independently: an error in one must not
   blank the ones after it. Failures are logged, never swallowed silently. */
function section(name, fn) {
  try { fn(); } catch (err) { console.error(`[home] section "${name}" failed:`, err); }
}

async function main() {
  let meta, examples;
  try {
    [meta, examples] = await Promise.all([
      fetch('data/meta.json').then(r => r.json()),
      fetch('data/examples.json').then(r => r.ok ? r.json() : []).catch(() => []),
    ]);
  } catch (e) {
    document.body.insertBefore(
      h('div', { class: 'empty' },
        'Could not load data/meta.json. Run build_data.py, then serve this folder over HTTP.'),
      document.body.firstChild);
    return;
  }

  const envColor = (id) => {
    const e = meta.envs.find(x => x.id === id);
    return e ? (isDark() ? e.color.dark : e.color.light) : CSS('--series-1');
  };
  const modelColor = (id) => {
    const m = meta.models.find(x => x.id === id);
    return m ? (isDark() ? m.color.dark : m.color.light) : CSS('--series-1');
  };
  const envLabel = (id) => (meta.envs.find(x => x.id === id) || {}).label || id;
  const modelLabel = (id) => (meta.models.find(x => x.id === id) || {}).label || id;

  /* ------------------------------------------------------------ hero stats */
  const S = meta.sample, C = meta.corpus, O = meta.overall;
  const P = C.paper || {};
  section('hero', () => {
    // Figures are quoted from the paper, not recomputed, so the two cannot drift.
    const tiles = [
      [P.scenarios, 'scenarios'],
      [P.sentences, 'sentences localized'],
      [P.continuations, 'sampled continuations'],
      [P.tokens, 'generated tokens'],
      [`${C.size_gb} GB`, 'compressed on the Hub'],
    ];
    $('hero-stats').append(...tiles.map(([v, l]) =>
      h('div', { class: 'stat' }, [h('div', { class: 'v' }, v), h('div', { class: 'l' }, l)])));
    $('hero-scale-note').innerHTML =
      `Every sentence boundary costs a fresh batch of continuations, so the corpus is far `
      + `larger than the traces it came from: ${P.scenarios} scenarios across `
      + `${C.n_envs} environments and ${C.n_models} reasoning models. Counts are as reported in `
      + `the paper (temporarily removed); `
      + `the compressed size is measured from the Hugging Face file listing.`;
  });

  /* -------------------------------------------------------------- pipeline */
  section('pipeline', () => {
    // The paper's own Figure 1 says this better than a redrawn version would.
    // It is inlined rather than used as <img> because the LaTeX export puts its
    // labels in <foreignObject>, which does not render inside an image element.
    fetch('assets/figure1.svg')
      .then(r => r.ok ? r.text() : Promise.reject(new Error(r.status)))
      .then((svg) => {
        const el2 = $('fig1');
        if (el2) el2.innerHTML = svg.replace(/^<\?xml[^>]*\?>\s*/, '');
      })
      .catch((err) => {
        const el2 = $('fig1');
        if (el2) el2.innerHTML =
          '<p style="color:var(--text-muted);font-size:13px">Figure 1 could not be '
          + 'loaded. It is in the paper (temporarily removed).</p>';
        console.warn('[home] figure 1:', err.message);
      });
  });

  /* ---------------------------------------------------------- environments */
  section('environments', () => {
    $('env-cards').append(...meta.envs.map(e => {
      const c = isDark() ? e.color.dark : e.color.light;
      return h('div', { class: 'card' }, [
        h('div', { style: 'display:flex;align-items:flex-start;gap:9px;margin-bottom:6px' }, [
          // sits on the first line of a wrapping title, not centred against it
          h('span', { style: `width:9px;height:9px;background:${c};flex:none;margin-top:7px` }),
          h('h3', {}, e.label),
        ]),
        h('div', { class: 'cap', style: 'font-family:var(--mono);font-size:11.5px;margin-bottom:8px' }, e.id),
        h('p', { style: 'margin:0;font-size:13.5px;color:var(--text-secondary)' }, e.blurb),
      ]);
    }));
  });

  /* ------------------------------------------------------------- examples */
  section('examples', () => {
    if (examples && examples.length) {
      $('examples').append(...examples.map(ex => exampleCard(ex, envColor, envLabel, modelLabel)));
    } else {
      $('examples').append(h('div', { class: 'empty' },
        'Examples appear once build_data.py has written data/examples.json.'));
    }
  });

}


function wrapText(g, text, x, y, maxw, size, fill) {
  const words = text.split(' ');
  let line = [], ly = y;
  const approx = size * 0.53;
  for (const w of words) {
    if ((line.join(' ') + ' ' + w).length * approx > maxw && line.length) {
      g.appendChild(el('text', { x, y: ly, style: `font-size:${size}px;fill:${fill}` }, line.join(' ')));
      line = [w]; ly += size + 3;
    } else line.push(w);
  }
  if (line.length) g.appendChild(el('text', { x, y: ly, style: `font-size:${size}px;fill:${fill}` }, line.join(' ')));
}

/* --------------------------------------------------------- example cards */
function exampleCard(ex, envColor, envLabel, modelLabel) {
  const byIdx = new Map(ex.probes.filter(p => p.r != null).map(p => [p.i, p]));
  const frag = h('div', { class: 'figure', style: 'margin-bottom:16px' });

  frag.appendChild(h('div', { class: 'fig-head' }, [
    h('div', { style: 'display:flex;align-items:center;gap:9px;flex-wrap:wrap' }, [
      h('span', { style: `width:10px;height:10px;border-radius:3px;background:${envColor(ex.env)};flex:none` }),
      h('h3', {}, `${envLabel(ex.env)} · ${modelLabel(ex.model)}`),
      h('span', {
        class: 'pill ' + (ex.r1 >= 0.5 ? 'dec' : 'hon'),
      }, ex.r1 >= 0.5 ? 'ends deceptive' : 'ends honest'),
      ex.j != null
        ? h('span', { class: 'pill none' }, `juncture at sentence ${ex.j}`)
        : h('span', {
            class: 'pill none',
            title: 'The commitment juncture is defined as the onset of deception, so it does '
                 + 'not apply to a trace that resolves toward disclosure. The marked boundary '
                 + 'is where the rate crosses 0.5 on the way down.',
          }, 'no deception onset'),
    ]),
    h('p', { class: 'mono', style: 'font-size:11px;color:var(--text-muted);margin-top:5px' }, ex.id),
  ]));

  frag.appendChild(h('p', { style: 'margin:0 0 10px;font-size:13px;color:var(--text-secondary)' }, ex.note || ''));

  // tinted sentences
  const body = h('div', { class: 'trace-text' });
  ex.probes.forEach((p) => {
    const span = h('span', {
      class: 'sent' + (p.r == null ? ' unprobed' : ''),
      style: p.r != null ? `background:${rateTint(p.r)}` : '',
    }, p.s + ' ');
    if (p.r != null) {
      span.addEventListener('mousemove', (evt) => showTip(
        `<div class="t-head"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${rateColor(p.r)}"></span>Boundary ${p.i}</div>` +
        `<div class="t-row"><span>deception rate</span><b>${fmtPct(p.r, 1)}</b></div>` +
        `<div class="t-row"><span>95% CI</span><b>${fmtPct(p.lo, 1)} &ndash; ${fmtPct(p.hi, 1)}</b></div>` +
        `<div class="t-row"><span>continuations</span><b>${p.nt} honest / ${p.nv} graded</b></div>`, evt));
      span.addEventListener('mouseleave', hideTip);
    }
    body.appendChild(span);
  });
  frag.appendChild(body);

  // the curve
  const curveBox = h('div', { style: 'margin-top:14px' });
  frag.appendChild(curveBox);
  const draw = () => {
    const pts = ex.probes.filter(p => p.r != null)
      .map(p => ({ x: p.i, y: p.r, lo: p.lo, hi: p.hi, n: p.nv, s: p.s }));
    lineChart(curveBox, [{
      key: 'r', label: '', color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(), pts,
    }], {
      width: 1010, height: 170, yDomain: [0, 1],
      xLabel: 'sentence boundary', yLabel: 'deception rate',
      xFmt: (v) => String(Math.round(v)), rule: 0.5, ruleLabel: '0.5',
      vrule: ex.mark_i != null ? ex.mark_i : null, vruleLabel: ex.mark_label || '',
      directLabel: false, margin: { t: 18, r: 30, b: 36, l: 46 },
      tipFmt: (p) => `<div class="t-head">Boundary ${Math.round(p.x)}</div>` +
        `<div class="t-row"><span>deception rate</span><b>${fmtPct(p.y, 1)}</b></div>` +
        `<div class="t-row"><span>95% CI</span><b>${fmtPct(p.lo, 1)} &ndash; ${fmtPct(p.hi, 1)}</b></div>` +
        `<div class="t-row"><span>graded</span><b>${fmtNum(p.n)}</b></div>` +
        `<div class="t-sent">${escapeHtml(p.s).slice(0, 190)}</div>`,
    });
  };
  draw();
  redraws.push(draw);
  return frag;
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}


main();
