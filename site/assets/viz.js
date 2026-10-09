/* ===========================================================================
   Shared chart primitives. Hand-rolled inline SVG - no external libraries, so
   the site works offline and from file://. Mark specs follow the project's
   data-viz rules: 2px lines, >=8px markers, recessive grid, a 2px surface ring
   on overlapping marks, direct labels plus a legend for every multi-series
   chart, and a hover layer on every plot.
   =========================================================================== */

const SVG = 'http://www.w3.org/2000/svg';

export const fmtPct = (v, d = 0) => v == null ? '--' : (100 * v).toFixed(d) + '%';
export const fmtNum = (v) => v == null ? '--' : v.toLocaleString('en-US');
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function el(tag, attrs = {}, kids = []) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  for (const c of [].concat(kids)) n.appendChild(typeof c === 'string'
    ? document.createTextNode(c) : c);
  return n;
}

export function h(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  for (const k in attrs) {
    if (k === 'class') n.className = attrs[k];
    else if (k === 'html') n.innerHTML = attrs[k];
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  }
  for (const c of [].concat(kids)) {
    if (c == null) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

/* ------------------------------------------------------------------ tooltip */
let tipEl = null;
export function tip() {
  if (!tipEl) {
    tipEl = h('div', { class: 'tip' });
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
export function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add('on');
  moveTip(ev);
}
export function moveTip(ev) {
  const t = tip();
  if (!t.classList.contains('on')) return;
  const pad = 14, r = t.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  t.style.left = Math.max(8, x) + 'px';
  t.style.top = Math.max(8, y) + 'px';
}
export function hideTip() { tip().classList.remove('on'); }

/* --------------------------------------------------------- diverging colour */
/* Honest (blue) <-> deceptive (red) with a neutral gray midpoint at 0.5.
   Never a rainbow, and no hue at the midpoint. */
const RAMP_LIGHT = ['#1b3a78', '#2b55a6', '#8faee4', '#eff0f1', '#eda9a9', '#cb4f4f', '#8e2f2f'];
const RAMP_DARK  = ['#a8c2f0', '#5a88df', '#2e4e93', '#2a2b30', '#8e3535', '#d96b6b', '#f0a9a9'];

function lerpHex(a, b, t) {
  const p = (s) => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  const [r1, g1, b1] = p(a), [r2, g2, b2] = p(b);
  const q = (x, y) => Math.round(x + (y - x) * t).toString(16).padStart(2, '0');
  return '#' + q(r1, r2) + q(g1, g2) + q(b1, b2);
}

export function isDark() {
  const s = document.documentElement.getAttribute('data-theme');
  if (s === 'dark') return true;
  if (s === 'light') return false;
  return matchMedia('(prefers-color-scheme: dark)').matches;
}

/** Deception rate -> diverging colour. 0 = honest pole, 0.5 = neutral, 1 = deceptive. */
export function rateColor(r) {
  if (r == null) return 'var(--surface-3)';
  const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
  const t = clamp(r, 0, 1) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(t));
  return lerpHex(ramp[i], ramp[i + 1], t - i);
}

/* Sequential blue ramp, for magnitudes that have no polarity (a share, a count).
   Light surface runs light->dark; the dark surface reverses it so "near zero"
   still recedes toward the surface rather than glowing. */
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];

export function seqColor(t) {
  const ramp = isDark() ? [...SEQ].reverse() : SEQ;
  const u = clamp(t, 0, 1) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(u));
  return lerpHex(ramp[i], ramp[i + 1], u - i);
}

/** Ink that stays legible on top of seqColor(t). */
export function seqInk(t) {
  const deep = isDark() ? clamp(t, 0, 1) < 0.45 : clamp(t, 0, 1) > 0.5;
  return deep ? '#ffffff' : '#0b0b0b';
}

/** Low-saturation version for tinting body text backgrounds. */
export function rateTint(r) {
  if (r == null) return 'transparent';
  const d = isDark();
  const a = Math.abs(r - 0.5) * 2;              // 0 at neutral, 1 at a pole
  const hue = r >= 0.5 ? (d ? '217,107,107' : '203,79,79')
                       : (d ? '90,136,223' : '62,110,205');
  return `rgba(${hue}, ${(a * (d ? 0.42 : 0.30)).toFixed(3)})`;
}

/* ------------------------------------------------------------------- scales */
export const scale = (d0, d1, r0, r1) => (v) =>
  d1 === d0 ? r0 : r0 + (r1 - r0) * ((v - d0) / (d1 - d0));

/* ------------------------------------------------------------------ helpers */
function axes(g, { x0, x1, y0, y1, yTicks, xTicks, sy, sx, yFmt, xFmt }) {
  for (const t of yTicks) {
    const y = sy(t);
    g.appendChild(el('line', { class: 'grid-line', x1: x0, x2: x1, y1: y, y2: y }));
    g.appendChild(el('text', {
      class: 'tick', x: x0 - 8, y: y + 3.5, 'text-anchor': 'end',
      style: 'font-size:11px;fill:var(--text-muted)'
    }, yFmt(t)));
  }
  g.appendChild(el('line', { class: 'axis-line', x1: x0, x2: x1, y1: y1, y2: y1 }));
  for (const t of xTicks) {
    const x = sx(t);
    g.appendChild(el('text', {
      class: 'tick', x, y: y1 + 16, 'text-anchor': 'middle',
      style: 'font-size:11px;fill:var(--text-muted)'
    }, xFmt(t)));
  }
}

function pathOf(pts) {
  return pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' ');
}

/* ===========================================================================
   lineChart - one or more series over a shared x, each with an optional
   confidence band. Always ships a crosshair + tooltip.
   series: [{ key, label, color, pts:[{x,y,lo,hi,n}] }]
   =========================================================================== */
export function lineChart(node, series, opts = {}) {
  // a caller may target a container that no longer exists (an older cached
  // script against newer markup); draw nothing rather than taking down the page
  if (!node) return null;
  const {
    width = 640, height = 260, xDomain, yDomain = [0, 1],
    xLabel = '', yLabel = '', yFmt = (v) => fmtPct(v), xFmt = (v) => String(v),
    xTicks, yTicks = [0, 0.25, 0.5, 0.75, 1], rule = null, ruleLabel = '',
    vrule = null, vruleLabel = '', directLabel = true, tipFmt = null,
    onPick = null,
    margin = { t: 12, r: 74, b: 34, l: 46 },
  } = opts;

  node.innerHTML = '';
  const all = series.flatMap(s => s.pts);
  if (!all.length) { node.appendChild(h('div', { class: 'empty' }, 'No data')); return; }

  const xd = xDomain || [Math.min(...all.map(p => p.x)), Math.max(...all.map(p => p.x))];
  const x0 = margin.l, x1 = width - margin.r, y0 = margin.t, y1 = height - margin.b;
  const sx = scale(xd[0], xd[1], x0, x1);
  const sy = scale(yDomain[0], yDomain[1], y1, y0);

  const svg = el('svg', {
    class: 'chart', viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'xMidYMid meet', role: 'img'
  });
  const g = el('g');
  svg.appendChild(g);

  const xt = xTicks || (() => {
    const n = 5, out = [];
    for (let i = 0; i < n; i++) out.push(xd[0] + (xd[1] - xd[0]) * i / (n - 1));
    return out;
  })();
  axes(g, { x0, x1, y0, y1, yTicks, xTicks: xt, sy, sx, yFmt, xFmt });

  if (rule != null) {
    g.appendChild(el('line', { class: 'rule', x1: x0, x2: x1, y1: sy(rule), y2: sy(rule) }));
    if (ruleLabel) g.appendChild(el('text', {
      x: x1 + 6, y: sy(rule) + 3.5, style: 'font-size:11px;fill:var(--text-muted)'
    }, ruleLabel));
  }
  if (vrule != null) {
    // amber marks the juncture here exactly as it does in the paper's figure
    g.appendChild(el('line', {
      x1: sx(vrule), x2: sx(vrule), y1: y0, y2: y1,
      stroke: 'var(--accent)', 'stroke-width': 1.2, 'stroke-dasharray': '4 3',
    }));
    if (vruleLabel) g.appendChild(el('text', {
      x: sx(vrule), y: y0 - 2, 'text-anchor': 'middle',
      style: 'font-size:11px;font-weight:600;fill:var(--accent)'
    }, vruleLabel));
  }

  // confidence bands first, so lines sit above them
  for (const s of series) {
    const band = s.pts.filter(p => p.lo != null && p.hi != null);
    if (band.length < 2) continue;
    const up = band.map(p => [sx(p.x), sy(clamp(p.hi, yDomain[0], yDomain[1]))]);
    const dn = band.map(p => [sx(p.x), sy(clamp(p.lo, yDomain[0], yDomain[1]))]).reverse();
    g.appendChild(el('path', {
      d: pathOf(up) + ' ' + pathOf(dn).replace('M', 'L') + ' Z',
      fill: s.color, opacity: 0.13, stroke: 'none'
    }));
  }

  for (const s of series) {
    g.appendChild(el('path', {
      d: pathOf(s.pts.map(p => [sx(p.x), sy(clamp(p.y, yDomain[0], yDomain[1]))])),
      fill: 'none', stroke: s.color, 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round'
    }));
    if (directLabel && s.label) {
      const last = s.pts[s.pts.length - 1];
      g.appendChild(el('text', {
        class: 'direct-label', x: sx(last.x) + 8,
        y: sy(clamp(last.y, yDomain[0], yDomain[1])) + 4, fill: s.color
      }, s.label));
    }
  }

  if (yLabel) g.appendChild(el('text', {
    class: 'axis-label', transform: `rotate(-90) translate(${-(y0 + y1) / 2} 13)`,
    'text-anchor': 'middle'
  }, yLabel));
  if (xLabel) g.appendChild(el('text', {
    class: 'axis-label', x: (x0 + x1) / 2, y: height - 2, 'text-anchor': 'middle'
  }, xLabel));

  // ---- hover layer: crosshair + nearest-x tooltip
  const cross = el('line', { class: 'axis-line', y1: y0, y2: y1, opacity: 0, 'stroke-dasharray': '3 3' });
  g.appendChild(cross);
  const dots = series.map(s => {
    const c = el('circle', { r: 4.5, fill: s.color, stroke: 'var(--surface-0)', 'stroke-width': 2, opacity: 0 });
    g.appendChild(c);
    return c;
  });
  const hit = el('rect', {
    x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0),
    fill: 'transparent', style: onPick ? 'cursor:pointer' : 'cursor:crosshair',
  });
  g.appendChild(hit);
  if (onPick) {
    hit.addEventListener('click', (ev) => {
      const box = svg.getBoundingClientRect();
      const px = (ev.clientX - box.left) / box.width * width;
      const xv = xd[0] + (xd[1] - xd[0]) * clamp((px - x0) / (x1 - x0), 0, 1);
      let bp = null;
      for (const s2 of series) for (const p of s2.pts) {
        if (!bp || Math.abs(p.x - xv) < Math.abs(bp.x - xv)) bp = p;
      }
      if (bp) onPick(bp);
    });
  }

  const onMove = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * width;
    const xv = xd[0] + (xd[1] - xd[0]) * clamp((px - x0) / (x1 - x0), 0, 1);
    let best = null;
    series.forEach((s, i) => {
      let bp = null;
      for (const p of s.pts) if (!bp || Math.abs(p.x - xv) < Math.abs(bp.x - xv)) bp = p;
      if (!bp) { dots[i].setAttribute('opacity', 0); return; }
      dots[i].setAttribute('cx', sx(bp.x));
      dots[i].setAttribute('cy', sy(clamp(bp.y, yDomain[0], yDomain[1])));
      dots[i].setAttribute('opacity', 1);
      if (!best || Math.abs(bp.x - xv) < Math.abs(best.p.x - xv)) best = { s, p: bp };
    });
    if (!best) return;
    cross.setAttribute('x1', sx(best.p.x));
    cross.setAttribute('x2', sx(best.p.x));
    cross.setAttribute('opacity', 0.5);
    showTip(tipFmt ? tipFmt(best.p, series) : defaultTip(best.p, series, xFmt, yFmt), ev);
  };
  hit.addEventListener('mousemove', onMove);
  hit.addEventListener('mouseleave', () => {
    hideTip(); cross.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
  });

  function defaultTip(p, ss, xf, yf) {
    let out = `<div class="t-head">${xLabel || 'x'} ${xf(p.x)}</div>`;
    for (const s of ss) {
      let bp = null;
      for (const q of s.pts) if (!bp || Math.abs(q.x - p.x) < Math.abs(bp.x - p.x)) bp = q;
      if (!bp) continue;
      out += `<div class="t-row"><span><span class="legend-dot" style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${s.color};margin-right:6px"></span>${s.label}</span><b>${yf(bp.y)}</b></div>`;
      if (bp.n) out += `<div class="t-row"><span style="opacity:.7">n</span><b>${fmtNum(bp.n)}</b></div>`;
    }
    return out;
  }

  node.appendChild(svg);
  return svg;
}

/* ===========================================================================
   barChart - horizontal bars with 4px rounded data-ends anchored to the
   baseline, an optional CI whisker, and a per-mark hover tooltip.
   =========================================================================== */
export function barChart(node, rows, opts = {}) {
  // a caller may target a container that no longer exists (an older cached
  // script against newer markup); draw nothing rather than taking down the page
  if (!node) return null;
  const {
    width = 640, rowH = 30, xDomain = [0, 1], xLabel = '',
    xFmt = (v) => fmtPct(v), valueLabel = true, tipFmt = null,
    margin = { t: 8, r: 60, b: 30, l: 150 },
  } = opts;

  node.innerHTML = '';
  if (!rows.length) { node.appendChild(h('div', { class: 'empty' }, 'No data')); return; }
  const height = margin.t + margin.b + rows.length * rowH;
  const x0 = margin.l, x1 = width - margin.r;
  const sx = scale(xDomain[0], xDomain[1], x0, x1);

  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${width} ${height}`, role: 'img' });
  const g = el('g'); svg.appendChild(g);

  for (const t of [0, 0.25, 0.5, 0.75, 1].map(f => xDomain[0] + f * (xDomain[1] - xDomain[0]))) {
    g.appendChild(el('line', { class: 'grid-line', x1: sx(t), x2: sx(t), y1: margin.t, y2: height - margin.b }));
    g.appendChild(el('text', {
      x: sx(t), y: height - margin.b + 15, 'text-anchor': 'middle',
      style: 'font-size:11px;fill:var(--text-muted)'
    }, xFmt(t)));
  }

  rows.forEach((r, i) => {
    const y = margin.t + i * rowH, bh = Math.min(17, rowH - 11);
    const cy = y + rowH / 2;
    g.appendChild(el('text', {
      x: x0 - 10, y: cy + 4, 'text-anchor': 'end',
      style: 'font-size:12.5px;fill:var(--text-primary)'
    }, r.label));
    const w = Math.max(2, sx(r.value) - x0);
    const rect = el('rect', {
      x: x0, y: cy - bh / 2, width: w, height: bh,
      rx: 4, fill: r.color || 'var(--series-1)', style: 'cursor:pointer'
    });
    g.appendChild(rect);
    // square off the baseline end so the bar is anchored, not floating
    g.appendChild(el('rect', {
      x: x0, y: cy - bh / 2, width: Math.min(4, w), height: bh,
      fill: r.color || 'var(--series-1)'
    }));

    if (r.lo != null && r.hi != null) {
      g.appendChild(el('line', {
        x1: sx(r.lo), x2: sx(r.hi), y1: cy, y2: cy,
        stroke: 'var(--text-primary)', 'stroke-width': 1.5, opacity: .55
      }));
      for (const b of [r.lo, r.hi]) g.appendChild(el('line', {
        x1: sx(b), x2: sx(b), y1: cy - 4, y2: cy + 4,
        stroke: 'var(--text-primary)', 'stroke-width': 1.5, opacity: .55
      }));
    }
    if (valueLabel) g.appendChild(el('text', {
      x: sx(r.hi != null ? r.hi : r.value) + 8, y: cy + 4,
      style: 'font-size:12px;font-weight:600;fill:var(--text-primary)'
    }, xFmt(r.value)));

    const hit = el('rect', { x: x0, y, width: x1 - x0, height: rowH, fill: 'transparent', style: 'cursor:pointer' });
    hit.addEventListener('mousemove', (ev) => showTip(
      tipFmt ? tipFmt(r) :
        `<div class="t-head"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${r.color}"></span>${r.label}</div>` +
        `<div class="t-row"><span>value</span><b>${xFmt(r.value)}</b></div>` +
        (r.lo != null ? `<div class="t-row"><span>95% CI</span><b>${xFmt(r.lo)} - ${xFmt(r.hi)}</b></div>` : '') +
        (r.n ? `<div class="t-row"><span>traces</span><b>${fmtNum(r.n)}</b></div>` : ''), ev));
    hit.addEventListener('mouseleave', hideTip);
    g.appendChild(hit);
  });

  if (xLabel) g.appendChild(el('text', {
    class: 'axis-label', x: (x0 + x1) / 2, y: height - 1, 'text-anchor': 'middle'
  }, xLabel));

  node.appendChild(svg);
  return svg;
}

/* ===========================================================================
   histogram - vertical bars, 2px surface gap between neighbours.
   =========================================================================== */
export function histogram(node, hist, opts = {}) {
  // a caller may target a container that no longer exists (an older cached
  // script against newer markup); draw nothing rather than taking down the page
  if (!node) return null;
  const {
    width = 640, height = 190, color = 'var(--series-1)', xLabel = '', yLabel = 'traces',
    xFmt = (v) => v.toFixed(1), colorFn = null, rule = null, ruleLabel = '',
    margin = { t: 10, r: 12, b: 34, l: 46 },
  } = opts;

  node.innerHTML = '';
  const { edges, counts } = hist;
  const max = Math.max(1, ...counts);
  const x0 = margin.l, x1 = width - margin.r, y0 = margin.t, y1 = height - margin.b;
  const sx = scale(edges[0], edges[edges.length - 1], x0, x1);
  const sy = scale(0, max, y1, y0);

  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${width} ${height}`, role: 'img' });
  const g = el('g'); svg.appendChild(g);

  const yt = [0, Math.round(max / 2), max];
  for (const t of yt) {
    g.appendChild(el('line', { class: 'grid-line', x1: x0, x2: x1, y1: sy(t), y2: sy(t) }));
    g.appendChild(el('text', {
      x: x0 - 8, y: sy(t) + 3.5, 'text-anchor': 'end',
      style: 'font-size:11px;fill:var(--text-muted)'
    }, fmtNum(t)));
  }

  counts.forEach((c, i) => {
    const bx = sx(edges[i]), bw = Math.max(1, sx(edges[i + 1]) - bx - 2);
    const bh = Math.max(c > 0 ? 1.5 : 0, y1 - sy(c));
    if (bh <= 0) return;
    const mid = (edges[i] + edges[i + 1]) / 2;
    const rect = el('rect', {
      x: bx, y: y1 - bh, width: bw, height: bh, rx: Math.min(4, bw / 2),
      fill: colorFn ? colorFn(mid) : color, style: 'cursor:pointer'
    });
    g.appendChild(rect);
    // square the baseline end
    g.appendChild(el('rect', {
      x: bx, y: Math.max(y1 - 4, y1 - bh), width: bw, height: Math.min(4, bh),
      fill: colorFn ? colorFn(mid) : color
    }));
    const hit = el('rect', { x: bx - 1, y: y0, width: bw + 2, height: y1 - y0, fill: 'transparent' });
    hit.addEventListener('mousemove', (ev) => showTip(
      `<div class="t-head">${xFmt(edges[i])} - ${xFmt(edges[i + 1])}</div>` +
      `<div class="t-row"><span>traces</span><b>${fmtNum(c)}</b></div>` +
      `<div class="t-row"><span>share</span><b>${fmtPct(c / counts.reduce((a, b) => a + b, 0), 1)}</b></div>`, ev));
    hit.addEventListener('mouseleave', hideTip);
    g.appendChild(hit);
  });

  g.appendChild(el('line', { class: 'axis-line', x1: x0, x2: x1, y1: y1, y2: y1 }));
  for (let i = 0; i < edges.length; i += Math.ceil(edges.length / 6)) {
    g.appendChild(el('text', {
      x: sx(edges[i]), y: y1 + 16, 'text-anchor': 'middle',
      style: 'font-size:11px;fill:var(--text-muted)'
    }, xFmt(edges[i])));
  }
  if (rule != null) {
    g.appendChild(el('line', { class: 'rule', x1: sx(rule), x2: sx(rule), y1: y0, y2: y1 }));
    if (ruleLabel) g.appendChild(el('text', {
      x: sx(rule), y: y0 + 10, 'text-anchor': 'middle',
      style: 'font-size:10.5px;fill:var(--text-muted)'
    }, ruleLabel));
  }
  if (xLabel) g.appendChild(el('text', {
    class: 'axis-label', x: (x0 + x1) / 2, y: height - 1, 'text-anchor': 'middle'
  }, xLabel));
  if (yLabel) g.appendChild(el('text', {
    class: 'axis-label', transform: `rotate(-90) translate(${-(y0 + y1) / 2} 12)`,
    'text-anchor': 'middle'
  }, yLabel));

  node.appendChild(svg);
  return svg;
}

/* ===========================================================================
   heatmap - environment x model grid on the sequential/diverging ramp.
   =========================================================================== */
export function heatmap(node, { rows, cols, cell, rowLabel, colLabel, fmt, title }, opts = {}) {
  // a caller may target a container that no longer exists (an older cached
  // script against newer markup); draw nothing rather than taking down the page
  if (!node) return null;
  const { width = 640, cellH = 40, margin = { t: 44, r: 12, b: 12, l: 150 } } = opts;
  node.innerHTML = '';
  const cw = (width - margin.l - margin.r) / cols.length;
  const height = margin.t + margin.b + rows.length * cellH;
  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${width} ${height}`, role: 'img' });
  const g = el('g'); svg.appendChild(g);

  cols.forEach((c, j) => g.appendChild(el('text', {
    x: margin.l + cw * (j + 0.5), y: margin.t - 12, 'text-anchor': 'middle',
    style: 'font-size:11.5px;fill:var(--text-secondary);font-weight:560'
  }, colLabel(c))));

  rows.forEach((r, i) => {
    const y = margin.t + i * cellH;
    g.appendChild(el('text', {
      x: margin.l - 10, y: y + cellH / 2 + 4, 'text-anchor': 'end',
      style: 'font-size:12.5px;fill:var(--text-primary)'
    }, rowLabel(r)));
    cols.forEach((c, j) => {
      const d = cell(r, c);
      const x = margin.l + cw * j;
      if (!d) return;
      g.appendChild(el('rect', {
        x: x + 1, y: y + 1, width: cw - 2, height: cellH - 2, rx: 5,
        fill: d.color, style: 'cursor:pointer'
      }));
      g.appendChild(el('text', {
        x: x + cw / 2, y: y + cellH / 2 + 4, 'text-anchor': 'middle',
        style: `font-size:12.5px;font-weight:640;fill:${d.ink || 'var(--text-primary)'}`
      }, fmt(d.value)));
      const hit = el('rect', { x, y, width: cw, height: cellH, fill: 'transparent', style: 'cursor:pointer' });
      hit.addEventListener('mousemove', (ev) => showTip(d.tip, ev));
      hit.addEventListener('mouseleave', hideTip);
      g.appendChild(hit);
    });
  });

  if (title) g.appendChild(el('text', {
    x: margin.l, y: 14, style: 'font-size:12px;fill:var(--text-muted)'
  }, title));

  node.appendChild(svg);
  return svg;
}

/* ===========================================================================
   sparkline - tiny deception curve for a list row. No axes, no hover (the row
   itself carries the tooltip), just the shape plus a 0.5 reference.
   =========================================================================== */
export function sparkline(pts, { width = 132, height = 30 } = {}) {
  const svg = el('svg', {
    class: 'spark', viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: 'none'
  });
  if (!pts.length) return svg;
  const xs = pts.map(p => p.i), x0 = Math.min(...xs), x1 = Math.max(...xs);
  const sx = scale(x0, x1 || x0 + 1, 2, width - 2);
  const sy = scale(0, 1, height - 3, 3);
  svg.appendChild(el('line', {
    x1: 0, x2: width, y1: sy(0.5), y2: sy(0.5),
    stroke: 'var(--border-strong)', 'stroke-width': 1, 'stroke-dasharray': '2 2'
  }));
  const dd = pts.map(p => [sx(p.i), sy(p.r)]);
  svg.appendChild(el('path', {
    d: dd.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' '),
    fill: 'none', stroke: 'var(--text-secondary)', 'stroke-width': 1.5,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: .85
  }));
  const last = pts[pts.length - 1];
  svg.appendChild(el('circle', {
    cx: sx(last.i), cy: sy(last.r), r: 3,
    fill: rateColor(last.r), stroke: 'var(--surface-0)', 'stroke-width': 1.5
  }));
  return svg;
}

/* ------------------------------------------------------------ theme toggle */
export function initTheme() {
  const KEY = 'cdl-theme';
  // ?theme=dark|light pins the theme for a shared link; otherwise the viewer's
  // last choice, otherwise the OS setting.
  const forced = new URLSearchParams(location.search).get('theme');
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
  const initial = (forced === 'dark' || forced === 'light') ? forced : saved;
  if (initial === 'dark' || initial === 'light') {
    document.documentElement.setAttribute('data-theme', initial);
  }
  const btn = document.getElementById('theme-btn');
  if (!btn) return;
  const paint = () => { btn.textContent = isDark() ? 'Light' : 'Dark'; };
  paint();
  btn.addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
    paint();
    document.dispatchEvent(new CustomEvent('themechange'));
  });
}
