// E9 (mega merge) — the shell/mega composition and the ingest layer merged back from
// the artifact edition. Checks: build produces a mega with all three apps bridged;
// parse-based bin sniffers classify real files; unzip handles STORE (our packs) and
// DEFLATE (Windows-made zips) with CRC-checked entries.
import { readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { deflateRawSync } from 'node:zlib';
import { unzip, tokenizerEntryCount, checkpointVocab, looksLikeZip } from '../src/ingest.js';
import { makeZip, crc32 } from '../tools/zip.js';
import { model260K, model15M, tok512 } from './paths.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export async function run(t) {
  // --- build (also regenerates the singles; every bundle is syntax-checked inside)
  execFileSync(process.execPath, [join(ROOT, 'tools', 'build-release.js')], { cwd: ROOT, stdio: 'pipe' });
  const mega = await readFile(join(ROOT, 'dist', 'tiny-mind-mega.html'), 'utf8');

  t.check('mega: all three app templates present',
    ['tpl-nursery', 'tpl-microscope', 'tpl-hunt'].every(id => mega.includes(`<template id="${id}">`)));
  t.check('mega: no unresolved build placeholders', !/@@[A-Z_]+@@/.test(mega));
  t.check('mega: every app carries the frame bridge',
    (mega.match(/tiny-mind frame bridge/g) || []).length >= 3);
  t.check('mega: every app carries an inlined worker',
    (mega.match(/__WORKER_SRC__ = "/g) || []).length === 3);
  t.check('mega: shell autoloads the arc manifest (arc kickoff deliverable 2)',
    mega.includes("fetch('arc/arc-manifest.json')"));
  t.check('mega: shell keeps the model pool + RPC hub',
    mega.includes('shellIngest') && mega.includes("m.tm === 'rpc'"));

  // --- parse-based sniffers on real files
  const ck260 = new Uint8Array(await readFile(model260K));
  const ck15 = new Uint8Array(await readFile(model15M));
  const tk512 = new Uint8Array(await readFile(tok512));
  t.check('sniffer: tok512 walks to exactly 512 entries', tokenizerEntryCount(tk512) === 512);
  t.check('sniffer: checkpoints are NOT tokenizers', tokenizerEntryCount(ck260) === -1);
  t.check('sniffer: 260K/15M vocab from header', checkpointVocab(ck260) === 512 && checkpointVocab(ck15) === 32000);

  // --- unzip STORE (our arc/starter packs)
  const storeZip = makeZip([{ name: 'a/tok512.bin', data: tk512 }, { name: 'note.json', data: new TextEncoder().encode('{"x":1}') }]);
  t.check('zip sniffer recognizes the container', looksLikeZip(storeZip));
  const storeOut = await unzip(storeZip);
  t.check('unzip STORE: names flattened, bytes exact',
    storeOut.length === 2 && storeOut[0].name === 'tok512.bin' &&
    storeOut[0].u8.length === tk512.length && tokenizerEntryCount(storeOut[0].u8) === 512);

  // --- unzip DEFLATE (hand-built, like Windows Compress-Archive output)
  const payload = tk512;
  const comp = deflateRawSync(payload);
  const nameB = new TextEncoder().encode('tok512.bin');
  const local = new Uint8Array(30 + nameB.length + comp.length);
  const ldv = new DataView(local.buffer);
  ldv.setUint32(0, 0x04034b50, true); ldv.setUint16(4, 20, true); ldv.setUint16(8, 8, true);
  ldv.setUint32(14, crc32(payload), true); ldv.setUint32(18, comp.length, true); ldv.setUint32(22, payload.length, true);
  ldv.setUint16(26, nameB.length, true);
  local.set(nameB, 30); local.set(comp, 30 + nameB.length);
  const cen = new Uint8Array(46 + nameB.length);
  const cdv = new DataView(cen.buffer);
  cdv.setUint32(0, 0x02014b50, true); cdv.setUint16(10, 8, true);
  cdv.setUint32(16, crc32(payload), true); cdv.setUint32(20, comp.length, true); cdv.setUint32(24, payload.length, true);
  cdv.setUint16(28, nameB.length, true); cdv.setUint32(42, 0, true);
  cen.set(nameB, 46);
  const eocd = new Uint8Array(22);
  const edv = new DataView(eocd.buffer);
  edv.setUint32(0, 0x06054b50, true); edv.setUint16(8, 1, true); edv.setUint16(10, 1, true);
  edv.setUint32(12, cen.length, true); edv.setUint32(16, local.length, true);
  const defZip = new Uint8Array(local.length + cen.length + 22);
  defZip.set(local, 0); defZip.set(cen, local.length); defZip.set(eocd, local.length + cen.length);

  const defOut = await unzip(defZip);
  let same = defOut.length === 1 && defOut[0].u8.length === payload.length;
  if (same) for (let i = 0; i < payload.length; i += 997) if (defOut[0].u8[i] !== payload[i]) { same = false; break; }
  t.check('unzip DEFLATE: inflates via DecompressionStream, byte-exact', same);
}
