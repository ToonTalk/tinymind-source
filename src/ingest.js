// ingest.js — universal drop/bundle ingestion, merged back from the mega/artifact
// edition (2026-07-06; three-phase core from the audit-v2 session, 2026-07-08).
// Shared by the Nursery, the shell, and tests. Bins are classified by PARSING, never
// by filename: a strict tokenizer-table walk to exact EOF, else a plausible llama2.c
// header. Zips inflate STORE directly and DEFLATE via the browser's
// DecompressionStream (also present in Node ≥18, so tests bite this).
import { splitArcPacks } from './scrubber.js';

export async function unzip(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let e = u8.length - 22;
  while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;   // EOCD signature
  if (e < 0) throw new Error('no zip end record');
  const count = dv.getUint16(e + 10, true);
  let off = dv.getUint32(e + 16, true);
  const out = [];
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(off, true) !== 0x02014b50) throw new Error('bad central directory');
    const method = dv.getUint16(off + 10, true);
    const csize = dv.getUint32(off + 20, true);
    const nameLen = dv.getUint16(off + 28, true), extraLen = dv.getUint16(off + 30, true), cmtLen = dv.getUint16(off + 32, true);
    const lho = dv.getUint32(off + 42, true);
    const name = new TextDecoder().decode(u8.subarray(off + 46, off + 46 + nameLen));
    if (!name.endsWith('/')) {
      const dataStart = lho + 30 + dv.getUint16(lho + 26, true) + dv.getUint16(lho + 28, true);
      const comp = u8.subarray(dataStart, dataStart + csize);
      let data;
      if (method === 0) data = comp.slice();
      else if (method === 8) data = new Uint8Array(await new Response(
        new Blob([comp]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer());
      else throw new Error('zip method ' + method + ' unsupported');
      out.push({ name: name.split('/').pop(), u8: data });
    }
    off += 46 + nameLen + extraLen + cmtLen;
  }
  return out;
}

export function looksLikeZip(u8) {
  return u8.length > 4 && u8[0] === 0x50 && u8[1] === 0x4b && u8[2] === 3;
}

// strict walk of [score f32][len i32][bytes]* to exact EOF -> vocab size, or -1
export function tokenizerEntryCount(u8) {
  if (u8.length < 12) return -1;
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let i = 4, n = 0;
  while (i + 8 <= u8.length) {
    const len = dv.getInt32(i + 4, true);
    if (len < 0 || len > 1024 || i + 8 + len > u8.length) return -1;
    i += 8 + len; n++;
  }
  return i === u8.length && n > 1 ? n : -1;
}

// llama2.c header sniff: [dim,hidden,layers,heads,kv,vocab,seq] -> |vocab|, or -1
export function checkpointVocab(u8) {
  if (u8.length < 28) return -1;
  const dv = new DataView(u8.buffer, u8.byteOffset, 28);
  const dim = dv.getInt32(0, true), layers = dv.getInt32(8, true), vocab = Math.abs(dv.getInt32(20, true));
  return dim > 0 && dim <= 16384 && layers > 0 && layers <= 256 && vocab > 1 ? vocab : -1;
}

// folder drop: FileSystemEntry -> [{name, u8}] (browser only; readEntries pages by 100)
export function walkEntry(entry) {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file(f => f.arrayBuffer().then(b => resolve([{ name: f.name, u8: new Uint8Array(b) }]), () => resolve([])), () => resolve([]));
    } else if (entry.isDirectory) {
      const reader = entry.createReader(), batches = [];
      const step = () => reader.readEntries(async (ents) => {
        if (!ents.length) resolve((await Promise.all(batches)).flat());
        else { for (const en of ents) batches.push(walkEntry(en)); step(); }
      }, () => resolve([]));
      step();
    } else resolve([]);
  });
}

// ---- three-phase ingest core (audit v2, Part 3) --------------------------------
// GATHER: expand every zip (recursively, depth-capped) into a flat record list —
// BEFORE any recognition, so an arc pack dropped as a .zip is visible to the
// splitter. SPLIT: arc packs out (manifest + the bins it names). CLASSIFY: the rest.
// DOM-free: callers bind handlers; the shell's drop path IS this function.
export async function expandZips(recs, { depth = 3, onError } = {}) {
  let out = [...recs];
  for (let d = 0; d < depth; d++) {
    let expanded = false;
    const next = [];
    for (const r of out) {
      if (/\.zip$/i.test(r.name) || looksLikeZip(r.u8)) {
        try { next.push(...await unzip(r.u8)); expanded = true; }
        catch (e) { if (onError) onError(r.name, e.message); }
        continue;
      }
      next.push(r);
    }
    out = next;
    if (!expanded) break;
  }
  return out;
}

export async function ingestBatch(recs, h) {
  recs = await expandZips(recs, { onError: h.unzipError });
  const packs = splitArcPacks(recs);
  for (const a of packs.arcs) h.takeArc(a.manifest, a.bins);
  let took = 0, jsons = 0;
  const skipped = [];
  for (const r of packs.rest) {
    if (/\.json$/i.test(r.name)) { jsons++; h.json(r.name, new TextDecoder().decode(r.u8)); continue; }
    const tc = tokenizerEntryCount(r.u8);
    if (tc > 0) { h.addTok(r.name, r.u8, tc); took++; continue; }
    const v = checkpointVocab(r.u8);
    if (v > 0) { h.addModel(r.name, r.u8, v); took++; continue; }
    skipped.push(r.name);
  }
  return { arcs: packs.arcs.map(a => a.manifest.arc), took, jsons, skipped };
}

// drop-event helper: snapshot BOTH views of the DataTransfer synchronously — it can be
// neutered at the first await, and webkitGetAsEntry() returns null after dispatch.
export async function collectDrop(e, onFolderStatus) {
  const entries = e.dataTransfer.items
    ? [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean) : [];
  const plainFiles = [...e.dataTransfer.files];
  if (entries.some(en => en.isDirectory)) {
    if (onFolderStatus) onFolderStatus();
    return (await Promise.all(entries.map(walkEntry))).flat();
  }
  return Promise.all(plainFiles.map(async f => ({ name: f.name, u8: new Uint8Array(await f.arrayBuffer()) })));
}
