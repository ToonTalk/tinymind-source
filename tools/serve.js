// serve.js — zero-dep static server for the E6 harness (node tools/serve.js [port]).
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, normalize, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.argv[2] ?? 8137);
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.bin': 'application/octet-stream',
  '.txt': 'text/plain', '.css': 'text/css',
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '');
    if (path === '' || path === '.') {
      res.writeHead(302, { location: '/microscope/index.html' });
      res.end();
      return;
    }
    if (url.pathname.endsWith('/')) path = join(path, 'index.html'); // directory index
    const full = join(ROOT, path);
    if (!full.startsWith(ROOT)) throw new Error('path escape');
    const data = await readFile(full);
    res.writeHead(200, {
      'content-type': MIME[extname(full)] ?? 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(data);
  } catch {
    res.writeHead(404); res.end('not found');
  }
}).listen(PORT, () => console.log(`harness at http://localhost:${PORT}/`));
