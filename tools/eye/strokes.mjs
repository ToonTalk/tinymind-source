// strokes.mjs — real Quick, Draw! strokes for the draw pad's test (Phase 1 of plans/tiny-eye.md).
// The 28x28 bitmaps the eye learned from were drawn from Quick, Draw!'s "simplified" strokes,
// in the same order. So the strokes of the sample doodles can go through the pad's own path
// (src/eye.js eyeDoodle) and be compared with the bitmaps the eye was trained on.
//   node tools/eye/strokes.mjs fetch [data folder]    first ~400 KB of each category's simplified
//                                                      .ndjson (Range request) -> <data>/strokes/
//   node tools/eye/strokes.mjs fixture [data folder]   sample doodles found there, up to 3 a
//                                                      category -> tests/fixtures/eye-strokes.json
//   node tools/eye/strokes.mjs check [data folder]     every sample doodle found: pad path vs bitmap
// Data: Quick, Draw! (Google, Inc., CC BY 4.0). The downloads are data, read as JSON lines only.
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const { TinyEye, eyeDoodle, eyeSampleInput } = await import(pathToFileURL(join(ROOT, 'src', 'eye.js')).href);
const [cmd, dataArg] = process.argv.slice(2);
const DATA = dataArg || 'C:/Users/toont/tm-data/eye';
const BASE = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/';
const BYTES = 400000;

const manifest = JSON.parse(readFileSync(join(ROOT, 'eye', 'tiny-eye.json'), 'utf8'));
const CLASSES = manifest.classes;

function sampleSet() {
  const u8 = readFileSync(join(ROOT, 'eye', manifest.sample.file));
  const src = JSON.parse(readFileSync(join(DATA, 'export', 'sample.json'), 'utf8'));
  return { u8, labels: src.labels, rows: src.source_row };
}
function lines(cat) {
  const f = join(DATA, 'strokes', cat + '.ndjson');
  return existsSync(f) ? readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];
}
// a Quick, Draw! drawing ([[xs], [ys]] per stroke) -> pad strokes ([[x, y], ...])
const padStrokes = d => d.drawing.map(([xs, ys]) => xs.map((x, i) => [x, ys[i]]));

if (cmd === 'fetch') {
  mkdirSync(join(DATA, 'strokes'), { recursive: true });
  for (const cat of CLASSES) {
    const r = await fetch(BASE + encodeURIComponent(cat) + '.ndjson', { headers: { Range: `bytes=0-${BYTES - 1}` } });
    if (r.status !== 206 && r.status !== 200) throw new Error(cat + ': HTTP ' + r.status);
    const text = await r.text();
    const whole = text.slice(0, text.lastIndexOf('\n') + 1);   // complete lines only
    writeFileSync(join(DATA, 'strokes', cat + '.ndjson'), whole);
    console.log(cat, whole.split('\n').length - 1, 'drawings');
  }
} else if (cmd === 'fixture' || cmd === 'check') {
  const S = sampleSet();
  const eye = new TinyEye(manifest, readFileSync(join(ROOT, 'eye', manifest.weights)));
  const found = [];
  for (let c = 0; c < CLASSES.length; c++) {
    const L = lines(CLASSES[c]);
    if (L.length && L[0].word !== CLASSES[c]) throw new Error(CLASSES[c] + ': the strokes file holds ' + L[0].word);
    for (let i = 0; i < S.labels.length; i++)
      if (S.labels[i] === c && S.rows[i] < L.length) found.push({ sample: i, label: c, row: S.rows[i], drawing: L[S.rows[i]].drawing, key: L[S.rows[i]].key_id });
  }
  const argmax = a => a.reduce((b, v, i) => v > a[b] ? i : b, 0);
  let sumCorr = 0, sumMad = 0, same = 0, rightPad = 0, rightBmp = 0;
  for (const f of found) {
    const pad = eyeDoodle(padStrokes(f));
    const bmp = eyeSampleInput(S.u8, f.sample);
    let ma = 0, mb = 0; for (let k = 0; k < 784; k++) { ma += pad[k]; mb += bmp[k]; } ma /= 784; mb /= 784;
    let sab = 0, saa = 0, sbb = 0, mad = 0;
    for (let k = 0; k < 784; k++) { const a = pad[k] - ma, b = bmp[k] - mb; sab += a * b; saa += a * a; sbb += b * b; mad += Math.abs(pad[k] - bmp[k]); }
    f.corr = sab / Math.sqrt(saa * sbb); f.mad = mad / 784;
    const gp = argmax(eye.forward(pad).logits), gb = argmax(eye.forward(bmp).logits);
    sumCorr += f.corr; sumMad += f.mad; same += gp === gb; rightPad += gp === f.label; rightBmp += gb === f.label;
  }
  const n = found.length;
  console.log(`${n} sample doodles with their strokes: pixel correlation ${(sumCorr / n).toFixed(4)} (worst ${Math.min(...found.map(f => f.corr)).toFixed(4)}), ` +
    `mean |difference| ${(sumMad / n).toFixed(4)}; the eye's guess agrees ${same}/${n}; right ${rightPad}/${n} through the pad, ${rightBmp}/${n} from the bitmaps`);
  if (cmd === 'fixture') {
    const keep = [];
    for (let c = 0; c < CLASSES.length; c++) keep.push(...found.filter(f => f.label === c).slice(0, 3));
    const out = { note: 'Quick, Draw! simplified strokes of sample doodles (eye/tiny-eye-sample.bin), from tools/eye/strokes.mjs. Data made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.',
      doodles: keep.map(f => ({ sample: f.sample, label: f.label, drawing: f.drawing })) };
    writeFileSync(join(ROOT, 'tests', 'fixtures', 'eye-strokes.json'), JSON.stringify(out));
    console.log(`tests/fixtures/eye-strokes.json: ${keep.length} doodles`);
  }
} else {
  console.error('usage: node tools/eye/strokes.mjs fetch|fixture|check [data folder]');
  process.exit(1);
}
