// storage.js — window.storage shim + chunked checkpoint persistence (kickoff §2, §6 E6).
// The real storage API (async get/set/delete, ~5MB per key) only exists inside claude.ai
// artifacts; this shim mimics it so the chunking layer can be built and tested outside.
// No localStorage anywhere (kickoff §10). DOM-free; Node + browser.

export function createMemoryStorageShim({ maxValueBytes = 5 * 1024 * 1024 } = {}) {
  const map = new Map();
  return {
    async get(key) { return map.has(key) ? map.get(key) : null; },
    async set(key, value) {
      if (typeof value !== 'string') throw new Error('storage shim: string values only');
      if (value.length > maxValueBytes) {
        throw new Error(`storage shim: value ${value.length} exceeds ${maxValueBytes} per key`);
      }
      map.set(key, value);
    },
    async delete(key) { map.delete(key); },
    _dump() { return map; },
  };
}

// base64 helpers that work in both Node and browser without Buffer
function bytesToBase64(bytes) {
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}
function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Save an ArrayBuffer under `${key}:meta` + `${key}:0..n-1` chunks (base64).
// chunkBytes counts base64 characters per key (default safely under the 5MB cap).
export async function saveCheckpointChunked(storage, key, arrayBuffer, chunkBytes = 4 * 1024 * 1024) {
  const b64 = bytesToBase64(new Uint8Array(arrayBuffer));
  const nChunks = Math.max(1, Math.ceil(b64.length / chunkBytes));
  // chunks first, meta last: meta is the commit marker, so a failed save is never
  // loadable (orphan chunks without meta are ignored by load and overwritten by retry)
  for (let i = 0; i < nChunks; i++) {
    await storage.set(`${key}:${i}`, b64.slice(i * chunkBytes, (i + 1) * chunkBytes));
  }
  await storage.set(`${key}:meta`, JSON.stringify({
    byteLength: arrayBuffer.byteLength, b64Length: b64.length, nChunks, v: 1,
  }));
  return { nChunks, b64Length: b64.length };
}

export async function loadCheckpointChunked(storage, key) {
  const metaRaw = await storage.get(`${key}:meta`);
  if (!metaRaw) return null;
  const meta = JSON.parse(metaRaw);
  let b64 = '';
  for (let i = 0; i < meta.nChunks; i++) {
    const part = await storage.get(`${key}:${i}`);
    if (part === null) throw new Error(`chunk ${i}/${meta.nChunks} missing for ${key}`);
    b64 += part;
  }
  if (b64.length !== meta.b64Length) throw new Error('chunk length mismatch');
  const bytes = base64ToBytes(b64);
  if (bytes.byteLength !== meta.byteLength) throw new Error('byte length mismatch');
  return bytes.buffer;
}

export async function deleteCheckpointChunked(storage, key) {
  const metaRaw = await storage.get(`${key}:meta`);
  if (!metaRaw) return;
  const meta = JSON.parse(metaRaw);
  for (let i = 0; i < meta.nChunks; i++) await storage.delete(`${key}:${i}`);
  await storage.delete(`${key}:meta`);
}
