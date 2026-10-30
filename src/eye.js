// eye.js — the Tiny Eye (plans/tiny-eye.md): a tiny ConvNeXt-style doodle classifier,
// trained offline on 24 Quick, Draw! categories (tools/eye/train.py, eye80k: 82,704
// weights), run here in plain JS. Two jobs:
//   - TinyEye: the forward pass from tiny-eye.json + tiny-eye.bin, with a tap on every
//     neuron of every stage and a switch-off mask. The neurons are the MLP hidden channels
//     (fc1 outputs, before the GELU); a drawing's value for a neuron is its mean over the
//     grid, the number Phase 0 labeled. Switching a neuron off zeroes its GELU output,
//     as train.py's mask does. Training snapshots (float16) load the same way.
//   - eyeDoodle: pad strokes -> the 28x28 picture the eye was trained on, made the way
//     Quick, Draw!'s bitmaps were made from its simplified drawings: align to the top-left,
//     scale the longer side to 255, simplify (Ramer-Douglas-Peucker, epsilon 2), then
//     draw centered with round, anti-aliased 16-unit lines on a 304-unit canvas scaled to
//     28 px (the longer side spans about 26 pixels; strokes about 1.5 pixels wide).
// Top-level names start with eye/EYE/TinyEye: the release build concatenates modules.

export const EYE_FORMAT = 'tiny-eye-v1';
const EYE_LN_EPS = 1e-5;

// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): GELU is 0.5 x (1 + erf(x / sqrt 2))
function eyeErf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496735) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

// out[oi + r] += W[wo + r*cols ...] . x[xi ...] for r < rows: four rows at a time, so each
// input number is loaded once per four products (rows is a multiple of 4 here)
function eyeMatVecAdd(W, wo, x, xi, out, oi, rows, cols) {
  for (let r = 0; r < rows; r += 4) {
    const w0 = wo + r * cols, w1 = w0 + cols, w2 = w1 + cols, w3 = w2 + cols;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let i = 0; i < cols; i++) {
      const v = x[xi + i];
      a0 += v * W[w0 + i]; a1 += v * W[w1 + i]; a2 += v * W[w2 + i]; a3 += v * W[w3 + i];
    }
    out[oi + r] += a0; out[oi + r + 1] += a1; out[oi + r + 2] += a2; out[oi + r + 3] += a3;
  }
}

// the same for two input vectors at once (two grid places share every weight they load)
function eyeMatVecAdd2(W, x, xa, xb, out, oa, ob, rows, cols) {
  for (let r = 0; r < rows; r += 4) {
    const w0 = r * cols, w1 = w0 + cols, w2 = w1 + cols, w3 = w2 + cols;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (let i = 0; i < cols; i++) {
      const u = x[xa + i], v = x[xb + i];
      let w = W[w0 + i]; a0 += u * w; b0 += v * w;
      w = W[w1 + i]; a1 += u * w; b1 += v * w;
      w = W[w2 + i]; a2 += u * w; b2 += v * w;
      w = W[w3 + i]; a3 += u * w; b3 += v * w;
    }
    out[oa + r] += a0; out[oa + r + 1] += a1; out[oa + r + 2] += a2; out[oa + r + 3] += a3;
    out[ob + r] += b0; out[ob + r + 1] += b1; out[ob + r + 2] += b2; out[ob + r + 3] += b3;
  }
}

// LayerNorm in place over each run of c numbers (channels-last), from `src` into `dst`
function eyeLayerNorm(src, dst, n, c, w, b) {
  for (let p = 0; p < n; p++) {
    const o = p * c;
    let m = 0; for (let i = 0; i < c; i++) m += src[o + i];
    m /= c;
    let v = 0; for (let i = 0; i < c; i++) { const d = src[o + i] - m; v += d * d; }
    const r = 1 / Math.sqrt(v / c + EYE_LN_EPS);
    for (let i = 0; i < c; i++) dst[o + i] = (src[o + i] - m) * r * w[i] + b[i];
  }
}

// IEEE half-precision bytes (little-endian) -> Float32Array: the training snapshots are stored this way
export function eyeHalfToFloat(bytes) {
  const u8 = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
  const n = u8.length >> 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = u8[2 * i] | (u8[2 * i + 1] << 8), sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class TinyEye {
  // manifest: tiny-eye.json (parsed); weights: ArrayBuffer / Uint8Array of tiny-eye.bin (float32), or of
  // one training snapshot with opts.float16 (the same tensors in the same places, two bytes each)
  constructor(manifest, weights, opts = {}) {
    if (!manifest || !manifest.config || !Array.isArray(manifest.tensors)) throw new Error('not a Tiny Eye manifest');
    const u8 = ArrayBuffer.isView(weights) ? new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength) : new Uint8Array(weights);
    const f32 = opts.float16 ? eyeHalfToFloat(u8) : new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
    const T = {};
    for (const t of manifest.tensors) {
      if (t.offset + t.length > f32.length) throw new Error('the weights are shorter than the manifest says (' + t.name + ')');
      T[t.name] = Float64Array.from(f32.subarray(t.offset, t.offset + t.length));   // doubles: no float32 conversions in the loops
    }
    const { widths, hidden, depths } = manifest.config;
    if (depths.some(d => d !== 1)) throw new Error('this engine runs one block per stage');
    if ([...widths, ...hidden].some(w => w % 4)) throw new Error('widths and hidden sizes must be multiples of 4');
    this.manifest = manifest;
    this.classes = manifest.classes;
    this.widths = widths; this.hidden = hidden;
    this.T = T;
    // grid sizes: 28 -> 14 (stem), then 14 -> 7 -> 4 (padding 1 when the grid is odd)
    this.sizes = [14];
    this.pads = [0];
    for (let i = 1; i < widths.length; i++) {
      const s = this.sizes[i - 1], pad = s % 2;
      this.pads.push(pad); this.sizes.push(Math.floor((s + 2 * pad - 2) / 2) + 1);
    }
    // weights rearranged so the inner loops run over contiguous memory
    this.dwT = widths.map((c, i) => {                // [ch][7][7] -> [k][ch]
      const w = T[`stages.${i}.0.dw.weight`], out = new Float64Array(49 * c);
      for (let ch = 0; ch < c; ch++) for (let k = 0; k < 49; k++) out[k * c + ch] = w[ch * 49 + k];
      return out;
    });
    this.downT = widths.slice(1).map((cout, j) => {  // [o][i][ky][kx] -> [ky*2+kx][o][i]
      const cin = widths[j], w = T[`downs.${j}.1.weight`], out = new Float64Array(4 * cout * cin);
      for (let o = 0; o < cout; o++) for (let i = 0; i < cin; i++) for (let k = 0; k < 4; k++)
        out[(k * cout + o) * cin + i] = w[(o * cin + i) * 4 + k];
      return out;
    });
    // scratch, sized for the biggest stage
    let big = 0;
    for (let i = 0; i < widths.length; i++) big = Math.max(big, this.sizes[i] * this.sizes[i] * Math.max(widths[i], hidden[i]));
    this.x = new Float64Array(big); this.t = new Float64Array(big); this.h = new Float64Array(big);
    this.y = new Float64Array(big); this.v = new Float64Array(2 * Math.max(...widths));
  }

  get nClasses() { return this.classes.length; }

  // x: 784 numbers, row-major, ink 1 and paper 0. opts.off: [Uint8Array per stage] with
  // 1 = switched off. opts.maps: also return each stage's neurons at every grid place.
  // Returns { logits, probs, neurons: [Float32Array per stage], maps? }.
  forward(input, opts = {}) {
    const { widths, hidden, sizes, pads, T } = this;
    const off = opts.off || [];
    let x = this.x, t = this.t;
    // stem: 2x2 stride-2 conv, 1 -> w0, then LayerNorm over the channels
    const c0 = widths[0], S0 = sizes[0], sw = T['stem.0.weight'], sb = T['stem.0.bias'];
    for (let oy = 0; oy < S0; oy++) for (let ox = 0; ox < S0; ox++) {
      const i00 = input[(2 * oy) * 28 + 2 * ox], i01 = input[(2 * oy) * 28 + 2 * ox + 1];
      const i10 = input[(2 * oy + 1) * 28 + 2 * ox], i11 = input[(2 * oy + 1) * 28 + 2 * ox + 1];
      const o = (oy * S0 + ox) * c0;
      for (let ch = 0; ch < c0; ch++)
        t[o + ch] = sb[ch] + i00 * sw[ch * 4] + i01 * sw[ch * 4 + 1] + i10 * sw[ch * 4 + 2] + i11 * sw[ch * 4 + 3];
    }
    eyeLayerNorm(t, x, S0 * S0, c0, T['stem.1.ln.weight'], T['stem.1.ln.bias']);
    const neurons = [], maps = opts.maps ? [] : null;
    for (let s = 0; s < widths.length; s++) {
      const c = widths[s], H = hidden[s], S = sizes[s], n = S * S;
      if (s > 0) {                                   // LayerNorm, then 2x2 stride-2 conv (zero padding)
        const cin = widths[s - 1], Sin = sizes[s - 1], pad = pads[s], wT = this.downT[s - 1], db = T[`downs.${s - 1}.1.bias`];
        eyeLayerNorm(x, t, Sin * Sin, cin, T[`downs.${s - 1}.0.ln.weight`], T[`downs.${s - 1}.0.ln.bias`]);
        for (let oy = 0; oy < S; oy++) for (let ox = 0; ox < S; ox++) {
          const o = (oy * S + ox) * c;
          for (let co = 0; co < c; co++) x[o + co] = db[co];
          for (let k = 0; k < 4; k++) {
            const iy = 2 * oy + (k >> 1) - pad, ix = 2 * ox + (k & 1) - pad;
            if (iy < 0 || ix < 0 || iy >= Sin || ix >= Sin) continue;
            const ib = (iy * Sin + ix) * cin;
            eyeMatVecAdd(wT, k * c * cin, t, ib, x, o, c, cin);
          }
        }
      }
      // block: depthwise 7x7 conv (padding 3) -> LayerNorm -> fc1 (the neurons) -> GELU -> fc2 -> x gamma -> + x
      const dw = this.dwT[s], dbias = T[`stages.${s}.0.dw.bias`], y = this.y;
      for (let py = 0; py < S; py++) for (let px = 0; px < S; px++) {
        const o = (py * S + px) * c;
        for (let ch = 0; ch < c; ch++) y[o + ch] = dbias[ch];
        for (let ky = 0; ky < 7; ky++) {
          const iy = py + ky - 3; if (iy < 0 || iy >= S) continue;
          for (let kx = 0; kx < 7; kx++) {
            const ix = px + kx - 3; if (ix < 0 || ix >= S) continue;
            const ib = (iy * S + ix) * c, wb = (ky * 7 + kx) * c;
            for (let ch = 0; ch < c; ch++) y[o + ch] += x[ib + ch] * dw[wb + ch];
          }
        }
      }
      eyeLayerNorm(y, t, n, c, T[`stages.${s}.0.ln.weight`], T[`stages.${s}.0.ln.bias`]);
      const w1 = T[`stages.${s}.0.fc1.weight`], b1 = T[`stages.${s}.0.fc1.bias`];
      const w2 = T[`stages.${s}.0.fc2.weight`], b2 = T[`stages.${s}.0.fc2.bias`], g = T[`stages.${s}.0.gamma`];
      const h = this.h, mean = new Float32Array(H), mask = off[s];
      for (let p = 0; p < n; p++) for (let j = 0; j < H; j++) h[p * H + j] = b1[j];
      for (let p = 0; p < n; p += 2) {
        if (p + 1 < n) eyeMatVecAdd2(w1, t, p * c, (p + 1) * c, h, p * H, (p + 1) * H, H, c);
        else eyeMatVecAdd(w1, 0, t, p * c, h, p * H, H, c);
      }
      if (maps) maps.push(h.slice(0, n * H));
      for (let p = 0; p < n; p++) {
        const hb = p * H;
        for (let j = 0; j < H; j++) {
          const v = h[hb + j];
          mean[j] += v;
          h[hb + j] = mask && mask[j] ? 0 : 0.5 * v * (1 + eyeErf(v * Math.SQRT1_2));
        }
      }
      const v = this.v;                             // fc2 for two places at a time, into v, then x += gamma * v
      for (let p = 0; p < n; p += 2) {
        const two = p + 1 < n;
        for (let ch = 0; ch < c; ch++) { v[ch] = b2[ch]; v[c + ch] = b2[ch]; }
        if (two) eyeMatVecAdd2(w2, h, p * H, (p + 1) * H, v, 0, c, c, H);
        else eyeMatVecAdd(w2, 0, h, p * H, v, 0, c, H);
        for (let ch = 0; ch < c; ch++) x[p * c + ch] += g[ch] * v[ch];
        if (two) for (let ch = 0; ch < c; ch++) x[(p + 1) * c + ch] += g[ch] * v[c + ch];
      }
      for (let j = 0; j < H; j++) mean[j] /= n;
      neurons.push(mean);
    }
    // head: mean over the grid -> LayerNorm -> linear
    const cL = widths[widths.length - 1], SL = sizes[sizes.length - 1], nL = SL * SL;
    const pooled = new Float32Array(cL);
    for (let p = 0; p < nL; p++) for (let ch = 0; ch < cL; ch++) pooled[ch] += x[p * cL + ch];
    for (let ch = 0; ch < cL; ch++) pooled[ch] /= nL;
    const normed = new Float32Array(cL);
    eyeLayerNorm(pooled, normed, 1, cL, T['head_ln.weight'], T['head_ln.bias']);
    const hw = T['head.weight'], hb = T['head.bias'], K = this.classes.length;
    const logits = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      let acc = hb[k];
      for (let ch = 0; ch < cL; ch++) acc += normed[ch] * hw[k * cL + ch];
      logits[k] = acc;
    }
    const probs = new Float32Array(K);
    let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
    let z = 0; for (let k = 0; k < K; k++) { probs[k] = Math.exp(logits[k] - mx); z += probs[k]; }
    for (let k = 0; k < K; k++) probs[k] /= z;
    const out = { logits, probs, neurons };
    if (maps) out.maps = maps;
    return out;
  }
}

// a 784-number input from one 28x28 doodle of a uint8 sample set (ink 255)
export function eyeSampleInput(u8, i, out = new Float32Array(784)) {
  const o = i * 784;
  for (let k = 0; k < 784; k++) out[k] = u8[o + k] / 255;
  return out;
}

// ---- the draw pad's path: strokes -> the eye's 28x28 picture ---------------------------

// Ramer-Douglas-Peucker on one stroke ([[x, y], ...]), as Quick, Draw!'s simplification
function eyeSimplify(pts, eps) {
  if (pts.length < 3) return pts.slice();
  const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
    let far = -1, fd = eps;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = pts[i];
      const d = L > 0 ? Math.abs(dy * px - dx * py + bx * ay - by * ax) / L : Math.hypot(px - ax, py - ay);
      if (d > fd) { fd = d; far = i; }
    }
    if (far >= 0) { keep[far] = 1; stack.push([a, far], [far, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// strokes in any coordinates (y down) -> Quick, Draw! "simplified" strokes: aligned to the
// top-left, the longer side scaled to 255, points at most 1 unit apart, then simplified
// with epsilon 2 and rounded to whole units
export function eyeNormalizeStrokes(strokes) {
  const ss = (strokes || []).filter(s => s && s.length);
  if (!ss.length) return [];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of ss) for (const [x, y] of s) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const span = Math.max(x1 - x0, y1 - y0), k = span > 0 ? 255 / span : 1;
  return ss.map(s => {
    const dense = [];
    for (let i = 0; i < s.length; i++) {
      const x = (s[i][0] - x0) * k, y = (s[i][1] - y0) * k;
      if (i) {
        const [px, py] = dense[dense.length - 1], d = Math.hypot(x - px, y - py), m = Math.floor(d);
        for (let j = 1; j < m; j++) dense.push([px + (x - px) * j / m, py + (y - py) * j / m]);
      }
      dense.push([x, y]);
    }
    return eyeSimplify(dense, 2).map(([x, y]) => [Math.round(x), Math.round(y)]);
  });
}

// simplified strokes (0..255) -> 784 numbers, ink 1, the way the 28x28 bitmaps were drawn:
// centered on a 256-unit square, 24 units of padding each side, round 16-unit lines,
// anti-aliased (16 x 16 samples per pixel), each stroke laid over the ones before
const EYE_SUB = 16;
export function eyeRaster(simplified, out = new Float32Array(784)) {
  out.fill(0);
  const ss = simplified.filter(s => s.length);
  if (!ss.length) return out;
  let mx = 0, my = 0;
  for (const s of ss) for (const [x, y] of s) { mx = Math.max(mx, x); my = Math.max(my, y); }
  const scale = 28 / 304, ox = (256 - mx) / 2 + 24, oy = (256 - my) / 2 + 24;
  const r = 8 * scale, r2 = r * r, G = 28 * EYE_SUB;
  const mark = new Uint16Array(G * G), cnt = new Uint16Array(784), touched = [];
  ss.forEach((s, si) => {
    const id = si + 1;
    const pts = s.map(([x, y]) => [(x + ox) * scale, (y + oy) * scale]);
    const segs = pts.length === 1 ? [[pts[0], pts[0]]] : pts.slice(1).map((p, i) => [pts[i], p]);
    for (const [[ax, ay], [bx, by]] of segs) {
      const dx = bx - ax, dy = by - ay, LL = dx * dx + dy * dy;
      const gx0 = Math.max(0, Math.floor((Math.min(ax, bx) - r) * EYE_SUB)), gx1 = Math.min(G - 1, Math.ceil((Math.max(ax, bx) + r) * EYE_SUB));
      const gy0 = Math.max(0, Math.floor((Math.min(ay, by) - r) * EYE_SUB)), gy1 = Math.min(G - 1, Math.ceil((Math.max(ay, by) + r) * EYE_SUB));
      for (let gy = gy0; gy <= gy1; gy++) {
        const py = (gy + 0.5) / EYE_SUB;
        for (let gx = gx0; gx <= gx1; gx++) {
          const g = gy * G + gx;
          if (mark[g] === id) continue;
          const px = (gx + 0.5) / EYE_SUB;
          let u = LL > 0 ? ((px - ax) * dx + (py - ay) * dy) / LL : 0;
          u = u < 0 ? 0 : u > 1 ? 1 : u;
          const ex = ax + u * dx - px, ey = ay + u * dy - py;
          if (ex * ex + ey * ey > r2) continue;
          mark[g] = id;
          const pix = ((gy / EYE_SUB) | 0) * 28 + ((gx / EYE_SUB) | 0);
          if (!cnt[pix]) touched.push(pix);
          cnt[pix]++;
        }
      }
    }
    for (const pix of touched) { const a = cnt[pix] / (EYE_SUB * EYE_SUB); out[pix] = a + out[pix] * (1 - a); cnt[pix] = 0; }
    touched.length = 0;
  });
  for (let i = 0; i < 784; i++) out[i] = Math.round(out[i] * 255) / 255;   // the bitmaps are 8-bit
  return out;
}

// the whole pad path: what the eye sees of a drawing
export function eyeDoodle(strokes, out) {
  return eyeRaster(eyeNormalizeStrokes(strokes), out);
}
