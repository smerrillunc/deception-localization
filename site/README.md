# Counterfactual Deception Localization — dataset site

A two-tab static site for the
[`a91939448/deception-localization`](https://huggingface.co/datasets/a91939448/deception-localization)
dataset.

- Paper: temporarily removed
- Dataset: <https://huggingface.co/datasets/a91939448/deception-localization>

Scale, as reported in the paper (do not recompute these — the site quotes them
so the two cannot drift apart):

| | |
|---|---|
| scenarios | 100K+ |
| sentences localized | ~1.46M |
| sampled continuations | 94.1M |
| generated tokens | 91.5B |
| compressed on the Hub | 105.4 GB |

The first four are quoted from the abstract. The paper states no figure for size
on disk; 105.4 GB is summed from the Hub's own file listing for the release.
These live in `PAPER` / `HUB_SIZE_GB` at the top of `build_data.py` — edit there
if the paper's numbers change.

- **Home** — what the dataset is, how each example is built, the headline
  findings, the five environments, three worked examples, and the schema.
- **Explore** — filter a 5,000-trace stratified sample by environment, model,
  commitment juncture, outcome, final rate, jump size and length; open any trace
  to see its full reasoning tinted by deception rate, its per-boundary curve,
  and the sampled continuations around its juncture.

Everything is plain HTML/CSS/JS with no build step and no runtime dependencies.

### Continuations browser

Each trace's drawer has a per-boundary continuations browser. Pick any probed
boundary and see the continuations sampled from that frozen prefix, split into
**deceptive**, **honest**, and **could not be graded**, with the share bar and
the parse error for each ungraded one.

Each boundary shows **one tile per sampled continuation**, coloured by how it was
graded — red deceptive, blue honest, outlined grey ungradeable. The whole outcome
distribution is legible at a glance (3 red among 97 blue *is* a 3% rate), and
clicking a tile pins that continuation open on its own.

Each boundary is labelled with its **Prefix** — the text actually held fixed
before resampling. It shows collapsed as `… <closing sentence>`, and the ellipsis
is a button: clicking it opens the whole prefix, with the closing sentence
highlighted at the end so you can see where the freeze fell.

**Clicking a sentence opens its continuations inline, directly underneath it** —
the share bar, one tile per continuation, and every generation as a card, without
leaving the trace. Click the sentence again (or *close*) to collapse it; clicking
a tile inside the expansion scrolls to that continuation's card.

The standalone panel further down does the same job and stays in step with the
sentence you picked. It can also be driven by clicking a point on the
deception-rate curve, or a numbered boundary chip.

**Every continuation is shown, from the Hub.** Opening a trace fetches its
original `.json.gz` from the Hub and gunzips it in the browser with
`DecompressionStream`, so the panel lists the complete set — typically 50 or 100
continuations at every probed boundary, often more than a thousand per trace.
This happens automatically; there is nothing to click.

The repo bundles four continuations around each trace's juncture purely so the
panel has something to draw in the first instant. It is replaced as soon as the
full source lands, and is the fallback if the fetch fails or the browser has no
`DecompressionStream`. Shipping the full set in the repo instead would push the
payload from 65 MB past a gigabyte. Nothing is proxied; the Hub sends permissive
CORS headers, so this works from any static host.

Worth knowing when reading these: the deception rate is computed over the graded
continuations only, so the denominator matters. The rates themselves have always
come from the full sample at each boundary — `deception_rate`, `num_valid` and
`num_truthful` are read straight from the source file, never recomputed from a
subset. It is shown explicitly
everywhere — per boundary in the browser, and as *graded / probe* on every row
and in each trace's drawer.

## Running locally

```bash
python3 -m http.server 8000     # from this directory
# then open http://localhost:8000
```

Opening `index.html` directly from the filesystem will not work: the pages fetch
JSON from `data/`, which browsers block over `file://`.

## Where the numbers come from

The prebuilt `data/` index exists so the Explore list can filter 5,000 traces
without fetching 5,000 files. **It is only a stand-in.** Opening a trace fetches
that trace's own `.json.gz` from the Hub, and the drawer then re-renders from it
— the curve, the intervals, the per-boundary counts and every continuation. The
*live from Hugging Face* badge marks that state. Nothing in an open drawer is
computed from local data once the source has loaded.

## Cache busting

`index.html` and `explore.html` load their assets with a `?v=N` query string,
and the JS modules import each other the same way. **Bump every `?v=` together
when you change anything under `assets/`.** Browsers cache modules aggressively,
and a stale script paired with newer markup is the one failure mode that looks
like content silently vanishing:

```bash
# from this directory, bump v=2 -> v=3 everywhere
sed -i 's/?v=2/?v=3/g' index.html explore.html assets/home.js assets/explore.js
```

Two defences make that failure non-fatal if it happens anyway: the chart helpers
return early on a missing container instead of throwing, and each section of the
Home page renders inside its own `section()` guard, so one failure is logged to
the console and the remaining sections still draw.

## Deploying to GitHub Pages

GitHub Pages serves a branch's **root** or its **/docs** folder — never an
arbitrary subfolder — so `site/` cannot be published where it sits. It goes to
the root of a `gh-pages` branch instead.

**Name the remote explicitly.** `deploy.sh` takes the remote as a required argument,
so nothing is published to a default remote by accident.

```bash
git remote add pages git@github.com:<you>/<repo>.git   # once
git add site && git commit -m "Add dataset site"
./site/deploy.sh pages
```

Then once, in the repo settings: **Settings → Pages → Source: Deploy from a
branch → `gh-pages` / `(root)`**. The first build takes a minute or two, and the
site lands at `https://<you>.github.io/<repo>/`.

Every path in the site is relative, so it works under a project subpath.
`.nojekyll` is committed so Pages serves `data/` verbatim instead of running it
through Jekyll.

Re-run `./site/deploy.sh pages` after any change; `gh-pages` is generated output,
so the script force-pushes it.

## Regenerating `data/`

`data/` is generated and checked in so the site is self-contained. To rebuild it
you need the two harvest files produced from the Hub (see *Harvesting* below):

```bash
python3 build_data.py harvest.jsonl --detail detail.jsonl
```

This writes:

| Path | Contents |
|---|---|
| `data/meta.json` | Counts, aggregate curves, per-environment and per-model summaries |
| `data/index.json` | One compact row per trace — everything the filters need |
| `data/curves/<env>__<model>.json` | Per-boundary rates and sentences, lazy-loaded per cell |
| `data/detail/<env>__<model>.json` | Full trace text, pooled prompts, sampled continuations |

Total payload is about 70 MB, of which only `meta.json` + `index.json` (~2.4 MB)
load up front; the rest is fetched per environment×model cell on demand.

## Harvesting

The site is built on a **stratified random sample of 5,000 traces** — 250 from
each of the 5 environments × 4 models — drawn with a fixed seed from the 99,999
in the full release. The full corpus is ~105 GB, which is why the site ships a
balanced sample rather than everything.

The traces are the ones already listed in `data/index.json`, so a rebuild shows the same
traces and only the underlying data changes. `harvest_local.py` reads them from a local
copy of the dataset and writes both inputs of `build_data.py`:

```bash
python3 harvest_local.py --root <dataset-root>     # <env>/<model>/localization/*.json.gz
python3 build_data.py harvest.jsonl --detail detail.jsonl
```

1. per-trace curves and metadata → `harvest.jsonl`
2. full trace text, prompt, and four sampled continuations (up to two deceptive) for the
   juncture boundary and its neighbouring probes → `detail.jsonl`

The site currently shows the **cleaned release (version 2)** of the dataset.

## Conventions

**Commitment juncture.** As the paper defines it: the first pair of **adjacent**
sentence boundaries — k and k+1, with no gap — whose counterfactual deception
rate shifts by at least **|Δp̂| = 0.30**, in *either* direction. A 30-point
collapse toward disclosure counts exactly as much as a 30-point jump toward
deception. `build_data.py` records the 0-based sentence closing the later
boundary, the direction (`jdir`: `rise` / `fall`) and the signed shift
(`jdelta`).

Adjacency is load-bearing. The corpus probes boundaries adaptively, so only 69%
of consecutive *probed* pairs are actually adjacent sentences; a Δ measured
across a multi-sentence gap is not a single-sentence shift, so those pairs are
skipped. The same adjacency rule applies to the *sharpest single-sentence jump*
filter in Explore, which would otherwise measure something its name does not
claim. A build-time check confirms no juncture sits on a non-adjacent pair.

Two earlier versions of this site got it wrong, and the numbers moved a long way
each time:

| rule | traces with a juncture | mean position |
|---|---|---|
| adaptive-search bracket (`right_sentence_end_idx`) — **wrong** | 79.4% | 0.35 |
| \|Δ\| ≥ 0.30 on consecutive *probed* boundaries — still wrong | 41.7% | 0.82 |
| **\|Δ\| ≥ 0.30 on adjacent sentences** | **35.9%** | **0.81** |

The search bracket is an artefact of how the corpus was probed, not a definition;
it must never stand in for one. Under the correct rule, junctures are much rarer
and much later in the trace than the site once claimed, and direction is almost
perfectly balanced (901 rises, 895 falls).

Because the rule is symmetric, averaging rates locked to the juncture cancels
out — the aligned mean is flat unless split by direction.

**Wording.** The site says *deceive* / *deception* throughout, never *lie*. The
environments define deception from state (a claim that contradicts hidden state,
a recommendation that is not on a shortest path, a non-disclosure when
disclosure is warranted), which is not the same claim as an intent to lie.

**Look.** The design is a technical monograph, not a landing page, and the
identity is derived from the data rather than applied on top of it.

*Type.* Newsreader for display and captions, IBM Plex Sans for body, IBM Plex
Mono for labels, ids and data. This is the one external dependency on the page —
Google Fonts — chosen because system-font stacks were the main reason earlier
versions read as generic. Every stack has a real fallback, so the site degrades
to Georgia/system sans offline rather than breaking.

*Structure.* **One grid for the whole page.** The hero and every section share a
single asymmetric measure — a `--rail` (148px) marker column and one content
column — so the title, prose, paper callout, statistics and figures all start and
end on the same two edges. Prose additionally holds `--measure` (64ch) for
readability, but shares its left edge with everything else; nothing sits at its
own arbitrary width. Earlier revisions had five competing max-widths stacked in
the hero alone, which is what made the page look ragged. Sections use that
measure as: the numeral sits out
in a 148px margin rail under an ink rule with its label beneath, and the prose
keeps a 58ch column beside it. The rail is why the page does not read as a stack
of identical bands. It collapses to one column below 860px. Numbering comes from
a CSS counter on `.eyebrow`, so a section without a marker does not consume a
number.

*Colour.* Colour belongs to the data. Prose links are ink with a hairline
underline rather than another blue competing with the honest/deceptive scale, and
the only chrome colour is the ink itself. The *commitment strip* — blocks holding
blue while continuations stay honest, then flipping red — runs as a 3px spine
across the top of every page and appears full size in the hero.

*Density.* Statistics are unboxed: Newsreader numerals on hairline rules, no
tiles. The Explore list is an index rather than a stack of cards — hairline rows
with a left spine tinted by how each trace ended, so scrolling shows the
distribution.

**Figure 1** is redrawn from the paper's TikZ source by `make_figure1.py`, not
lifted from the paper's HTML build (whose render of this figure is broken). The
generator maps TikZ centimetres to SVG user units and flips the y axis, so every
coordinate in the script is the one in the `.tex` — re-run it against an edited
figure and the output tracks. It is inlined by `home.js` and runs as a
**full-width plate**: at 3.5:1 it is unreadable inside the text column, and it is
the one element allowed to break the grid, which is what makes it read as a
plate. It sits outside `.wrap` so that width needs no viewport arithmetic — the
asymmetric grid makes the usual `calc(50% - 50vw)` trick overflow. It keeps a
white ground in both themes: a printed artefact, not a themed chart.

**Colour comes from the figure.** The site's honest/deceptive poles are the
paper's own `truth` (#3E6ECD) and `decep` (#CB4F4F), and amber (#B45309 light,
#E8991F dark) carries the role it has in Figure 1 — marking the fixed prefix, the
kept pair, the commitment juncture. On the site it marks section numerals, the
juncture rule in every chart, and active state. Amber is deliberately *not* a
data series: it sits only ΔE 7.8 from the red, which is fine for an annotation
carrying a label but would fail as a third series colour. The two poles validate
all-pairs (CVD ΔE 19.8, normal-vision ΔE 27.6, both ≥3:1 on the surface).

**Colour.** Deception rate is a polarity, so it uses a blue↔red diverging scale
with a neutral midpoint at 0.5. Commitment rate is a magnitude, so it uses a
single-hue sequential blue scale. Environments and models take fixed categorical
slots that are never recycled. The palette is colourblind-validated in both light
and dark themes; every multi-series chart carries a legend, direct labels, and a
table view.

**Data version.** The site is built from the cleaned release of the dataset (version 2). In
the first release some continuations were ungradeable because the stored text still held
raw byte-level BPE symbols (`Ġ` for a space, `Ċ` for a newline), which broke the JSON parse;
others were malformed in smaller ways. The cleaned release decodes the BPE text, repairs
near-valid JSON, extracts explicitly stated answers, and strips the gpt-oss channel markers
(`assistantfinal`, `assistantanalysis`, `assistantcommentary`) from generated text; each
recovered continuation is tagged in a `recovery` field in the data. Across the release,
99.1% of continuations are now graded (94.9% before). The deception rate of every affected
boundary was recomputed over the larger set of graded continuations, so some curves and
junctures differ from the first release.

**Known source artefact.** About 4% of stored continuations — concentrated in
`DeepSeek-R1-Distill-Llama-8B` and `gpt-oss-20b` — lost their word-separator
tokens upstream, so words run together. `build_data.py` decodes byte-level BPE
symbols where they survive and flags the rest in the UI rather than inventing
word boundaries. Labels and rates are unaffected.

**Worked examples.** The three traces on the Home tab are selected by
`build_data.py` under explicit constraints rather than hand-picked: the juncture
must come from the adaptive search (not the 0.5 fallback), sit between 0.15 and
0.85 of the way through the trace (the position distribution is strongly bimodal,
and traces committing at the very first or very last sentence do not illustrate
anything), average at least 25 graded continuations per boundary, and show a
clean flat-then-stepped shape. Environment and model are kept distinct across the
three.

**Sampling quality.** Grading is reliable: the median probed boundary has **50 graded
continuations**, matching the ~50 target, and 99.1% of sampled continuations are graded.
In the first release about 2.2% of the sample (112 of 5,000 traces) were outliers where
most continuations failed to parse, and they were over-represented among the sharpest
jumps (a rate estimated from ~5 samples lands on exactly 0 or 1 with ease). After cleaning,
no trace in the sample falls below 10 graded continuations per boundary.

The *Sampling quality* filter is kept: it defaults to requiring 10 graded continuations
per boundary on average, the sidebar states how many traces it holds back (currently
none), and the slider goes to 0. It affects Explore only; the Home aggregates use all 5,000.

**Sampling caveat.** Every figure on the site is an estimate from the 5,000-trace
sample, not a measurement of the full 99,999.

## Citation

Temporarily removed.
