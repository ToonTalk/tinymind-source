// E6 (partial, Node-side) — chunked window.storage persistence of a fine-tuned
// checkpoint through the shim. Browser timings run in harness/index.html; this test
// pins the chunking logic itself (split, reassemble, byte-exact, per-key cap).
import { readFile } from 'node:fs/promises';
import { createMemoryStorageShim, saveCheckpointChunked, loadCheckpointChunked, deleteCheckpointChunked }
  from '../harness/storage.js';
import { readCheckpoint } from '../src/bin-reader.js';
import { model260K } from './paths.js';

export async function run(t) {
  const buf = (await readFile(model260K)).buffer.slice(0); // ~1.06 MB
  const shim = createMemoryStorageShim();

  // small chunk size to force real chunking (260K b64 ≈ 1.41 MB -> 6 chunks @ 256KB)
  const saved = await saveCheckpointChunked(shim, 'ck', buf, 256 * 1024);
  t.check(`chunked save produced multiple chunks (${saved.nChunks})`, saved.nChunks >= 5);

  const back = await loadCheckpointChunked(shim, 'ck');
  const a = new Uint8Array(buf), b = new Uint8Array(back);
  let same = a.length === b.length;
  if (same) for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { same = false; break; }
  t.check('round-trip is byte-exact', same);

  const ck = readCheckpoint(back);
  t.check('restored buffer parses as a valid 260K checkpoint', ck.config.vocabSize === 512);

  // per-key cap enforced (5MB default): oversized single chunk must throw
  let threw = false;
  try { await saveCheckpointChunked(shim, 'big', new ArrayBuffer(6 * 1024 * 1024), 100 * 1024 * 1024); }
  catch { threw = true; }
  t.check('oversized single-key write rejected by 5MB shim cap', threw);

  await deleteCheckpointChunked(shim, 'ck');
  t.check('delete removes all keys', shim._dump().size === 0);

  // 5MB/key with 4MB chunks: a full 15M (~60MB fp32 -> ~81MB b64) would need ~21 keys —
  // fine for the API, but slow; the app only persists the 260K trainee (1 key at 4MB).
  console.log(`  260K checkpoint: ${(buf.byteLength / 1e6).toFixed(2)} MB -> ${saved.nChunks} chunks @256KB (test) / 1 chunk @4MB (app default)`);
}
