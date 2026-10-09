/* ===========================================================================
   On-demand loading of a trace's original source file from the Hugging Face
   Hub.

   The site bundles a small sample of continuations per trace. Shipping all ~50
   for every probed boundary would take the payload from 65 MB to well over a
   gigabyte, so the full set is fetched from the Hub when the reader asks for it
   and decompressed in the browser.

   The Hub sends `access-control-allow-origin` on the redirect and the CDN sends
   `*` on the object itself, so a plain cross-origin fetch works from a static
   site. The object is served as `application/gzip` without a
   `content-encoding` header, so the browser does not unzip it for us -
   DecompressionStream does.
   =========================================================================== */

const REPO_BASE =
  'https://huggingface.co/datasets/a91939448/deception-localization/resolve/main/';

export const canLoadSource = () => typeof DecompressionStream !== 'undefined';

/* ------------------------------------------------------- byte-level BPE ---- */
/* Some models store `gen_text` as raw byte-level BPE symbols (U+0120 for a
   space, U+010A for a newline). Undo the GPT-2 byte<->unicode map, mirroring
   fix_bpe() in build_data.py so bundled and freshly fetched text agree. */
const BYTE_DECODER = (() => {
  const bs = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const m = new Map();
  for (let i = 0; i < bs.length; i++) m.set(String.fromCharCode(cs[i]), bs[i]);
  return m;
})();

const UTF8 = new TextDecoder('utf-8');

export function fixBpe(text) {
  if (!text || (!text.includes('Ġ') && !text.includes('Ċ'))) return text;
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) {
    const b = BYTE_DECODER.get(text[i]);
    if (b === undefined) {
      // mixed encoding - fall back to swapping just the two common symbols
      return text.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
    }
    out[i] = b;
  }
  return UTF8.decode(out);
}

/** True when a decoded continuation still has its word separators missing. */
export function isLossy(text) {
  if (!text) return false;
  let run = 0;
  for (let i = 0; i < text.length; i++) {
    if (/\s/.test(text[i])) run = 0;
    else if (++run > 40) return true;
  }
  return false;
}

/* ------------------------------------------------------------- fetching ---- */
const cache = new Map();     // path -> normalised probes
const MAX_CACHED = 4;

/** True when a parse failure was caused by undecoded BPE rather than bad output. */
function recoverableJson(err) {
  if (!err || (!err.includes('Ġ') && !err.includes('Ċ'))) return false;
  const nl = err.indexOf('\n');
  if (nl < 0) return false;
  try { JSON.parse(fixBpe(err.slice(nl + 1))); return true; } catch (e) { return false; }
}

/** Environment metadata arrives as an object or a stringified dict; render both. */
function metaString(v) {
  if (v == null) return null;
  if (typeof v === 'string') return v;
  try { return JSON.stringify(v); } catch (e) { return String(v); }
}

function normalise(doc) {
  const out = new Map();
  for (const h of doc.history || []) {
    const gens = (h.generations || []).map((g, idx) => {
      const text = fixBpe(g.gen_text || '');
      // `deceptive` is the complement of `is_truthful` when grading succeeded;
      // when neither is set the continuation could not be graded at all.
      let label = 'ungraded';
      if (g.deceptive === true || g.is_truthful === false) label = 'deceptive';
      else if (g.is_truthful === true || g.deceptive === false) label = 'honest';
      const rawErr = g.parse_error == null ? null : String(g.parse_error);
      return {
        n: idx + 1,                       // position in the original sample
        text,
        label,
        lossy: isLossy(text),
        // The grader ran json.loads on text that still held byte-level BPE
        // symbols, so it rejected output that is valid JSON once decoded. Flag
        // those rather than relabelling them - applying the environment rule
        // here would mean reimplementing five graders.
        recoverable: label === 'ungraded' && recoverableJson(rawErr),
        err: rawErr == null ? null : fixBpe(rawErr),
        evaluation: metaString(g.evaluation),
        parsed: metaString(g.parsed),
      };
    });
    out.set(h.sentence_end_idx, {
      i: h.sentence_end_idx,
      sentence: h.sentence_text || '',
      prefix: h.prefix_text || '',
      rate: h.deception_rate,
      ciLow: h.ci_low,
      ciHigh: h.ci_high,
      numValid: h.num_valid || 0,
      numTruthful: h.num_truthful || 0,
      gens,
    });
  }
  return out;
}

/**
 * Fetch and decompress one trace's source file.
 * @param {string} path repo-relative path, as stored in the site index
 * @param {(loaded:number, total:number|null)=>void} [onProgress]
 */
export async function loadSource(path, onProgress) {
  if (cache.has(path)) return cache.get(path);
  if (!canLoadSource()) throw new Error('This browser cannot decompress gzip in-page.');

  const url = REPO_BASE + path.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Hugging Face returned ${res.status}`);

  const total = Number(res.headers.get('content-length')) || null;
  let stream = res.body;

  if (onProgress && stream) {
    let loaded = 0;
    stream = stream.pipeThrough(new TransformStream({
      transform(chunk, ctl) { loaded += chunk.byteLength; onProgress(loaded, total); ctl.enqueue(chunk); },
    }));
  }

  const text = await new Response(
    stream.pipeThrough(new DecompressionStream('gzip')),
  ).text();

  const probes = normalise(JSON.parse(text));
  if (cache.size >= MAX_CACHED) cache.delete(cache.keys().next().value);
  cache.set(path, probes);
  return probes;
}

export const isLoaded = (path) => cache.has(path);
export const peek = (path) => cache.get(path) || null;
