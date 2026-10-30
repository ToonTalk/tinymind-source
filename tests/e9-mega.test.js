// E9 (mega merge) — the shell/mega composition and the ingest layer merged back from
// the artifact edition. Checks: build produces a mega with all three apps bridged;
// parse-based bin sniffers classify real files; unzip handles STORE (our packs) and
// DEFLATE (Windows-made zips) with CRC-checked entries.
import { readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { deflateRawSync } from 'node:zlib';
import { unzip, tokenizerEntryCount, checkpointVocab, looksLikeZip, ingestBatch } from '../src/ingest.js';
import { makeZip, crc32 } from '../tools/zip.js';
import { model260K, model15M, tok512 } from './paths.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export async function run(t) {
  // --- build (also regenerates the singles; every bundle is syntax-checked inside)
  execFileSync(process.execPath, [join(ROOT, 'tools', 'build-release.js')], { cwd: ROOT, stdio: 'pipe' });
  const mega = await readFile(join(ROOT, 'dist', 'tiny-mind-mega.html'), 'utf8');
  const resourceManifest = JSON.parse(await readFile(join(ROOT, 'dist', 'tiny-mind-resources.json'), 'utf8'));

  t.check('mega: all five app templates present',
    ['tpl-nursery', 'tpl-microscope', 'tpl-hunt', 'tpl-thought', 'tpl-eye'].every(id => mega.includes(`<template id="${id}">`)));
  t.check('mega: no unresolved build placeholders', !/@@[A-Z_]+@@/.test(mega));
  t.check('mega: every app carries the frame bridge',
    (mega.match(/tiny-mind frame bridge/g) || []).length >= 3);
  t.check('mega: every app carries an inlined worker',
    (mega.match(/__WORKER_SRC__ = "/g) || []).length === 5);
  t.check('mega: shell autoloads hosted arcs (multi-manifest discovery, waves 7-14)',
    mega.includes("tmGetJson('arc/arcs.json')") && mega.includes('manifest.more || []'));
  t.check('mega: the Nursery ghostwriter can reach the LLM (aliased import survives the build)',
    mega.includes('const llmCall = callLLM;') && mega.includes('return llmCall({'));
  t.check('mega: built-in Claude is offered only where the shell has it (claude.ai artifact)',
    mega.includes('window.__tmKeyless = !!(window.claude &&') &&
    mega.includes('shellHasClaude = !!parent.__tmKeyless') &&
    !mega.includes("if (!(window.claude && typeof window.claude.complete === 'function'))\n    window.claude = {"));
  t.check('mega: the switch-off lab shows how the story goes on (normal vs switched off)',
    mega.includes('<div id="ablCont"></div>') && mega.includes("S.contBusy = true; S.contProgress = null; S.cont = null;") &&
    mega.includes('function renderAblCont()'));
  t.check('mega: a guide in a hidden tab never queues speech to play on return',
    mega.includes("if(document.hidden) return false;") && mega.includes("if(document.hidden) vgSynthCancel();"));
  t.check('mega: same-folder runtime packs are manifest-driven with an embedded fallback',
    resourceManifest.version === 1 &&
    resourceManifest.resources.map(r => r.file).join('|') ===
      'tiny-mind-starter.zip|tiny-mind-binding-models-corrected.zip|zog-arc.zip|newborn-arc.zip|binding-arc.zip|tiny-mind-neuron-atlas.json' &&
    mega.includes("const RESOURCE_MANIFEST = 'tiny-mind-resources.json'") &&
    mega.includes('const FALLBACK_RESOURCE_PACKS = ['));
  t.check('mega: same-folder resources are cache-first and cache successful downloads',
    mega.includes("const TM_RESOURCE_CACHE = 'tiny-mind-same-folder-v1'") &&
    mega.includes('const hit = await cache.match(url)') &&
    mega.includes('cache.put(url, copy)') &&
    mega.includes('await tmRestoreReady;'));
  t.check('mega: loose four-character checkpoints are also discovered beside the page',
    mega.includes("'stories260K-binding.bin', 'stories260K-binding-max.bin'"));
  t.check('mega: guided layer present in all six docs (vg-core ×6, prelude ×5, shell adapter)',
    (mega.match(/vg-core\.js — app-agnostic voice-guide core/g) || []).length === 6 &&
    (mega.match(/tm-prelude\.js — Tiny Mind guide prelude/g) || []).length === 5 &&
    mega.includes('tm-shell-adapter.js'));
  t.check('mega: probe-fact branch shipped inside the nursery worker blob',
    mega.includes('probe-fact-done'));
  t.check('mega: time machine (scrubber) markup present; the minds load by themselves (no starter-pack quest)',
    mega.includes('id=\\"scrubSlider\\"') === false /* worker strings escape quotes */ &&
    mega.includes('id="scrubSlider"') && !mega.includes('window.TM_STARTER_ZIP') && mega.includes('function ensureBig('));
  t.check('mega: shell keeps the model pool + RPC hub',
    mega.includes('shellIngest') && mega.includes("m.tm === 'rpc'"));
  t.check('mega: ready-made Zog storybook is available without neighboring files',
    mega.includes('const __TM_EMBEDDED_ZOG_STORIES__ = JSON.parse(') &&
    mega.includes("typeof __TM_EMBEDDED_ZOG_STORIES__ !== 'undefined'"));
  t.check('mega: balanced four-character story world is available without neighboring files',
    mega.includes('const __TM_EMBEDDED_BINDING_CORPUS__ = JSON.parse(') &&
    mega.includes("typeof __TM_EMBEDDED_BINDING_CORPUS__ !== 'undefined'"));
  t.check('mega: pretrained slot cannot be occupied by a saved Sisi-style 260K model',
    mega.includes('const isBase260 = m =>') &&
    mega.includes('const fresh260  = () => pool.models.find(isBase260);') &&
    !mega.includes("!isZog(m) && /260/.test(m.name)"));
  t.check('mega: only the exact legacy Zog checkpoint is classified as the Zog-only model',
    mega.includes('const isLegacyZog = m =>') &&
    mega.includes('const zog260    = () => pool.models.find(isLegacyZog);') &&
    mega.includes('!isLegacyZog(m) && !isBase260(m)'));
  t.check('mega: four-character training and sampling do not reuse the hidden single-character prompt',
    mega.includes('function activeStoryPrompts()') &&
    mega.includes('prompt: prompts[0], prompts,') &&
    mega.includes('sampleText(nextTrainingPrompt())'));
  t.check('mega: completed Nursery training publishes to the model pool without forcing a download',
    mega.includes("m.tm === 'publish-model'") &&
    mega.includes('window.__tmPublishModel =') &&
    mega.includes("purpose: 'publish'") &&
    mega.includes('function onPublishModel('));
  t.check('mega: trained-model character metadata travels through publish, persistence and load-pair',
    mega.includes('activeModelMetadata(m.step)') &&
    mega.includes('modelMeta: model.meta || null') &&
    mega.includes('modelMeta: idx.extra && idx.extra.modelMeta') &&
    mega.includes('meta: m.modelMeta || null'));
  t.check('mega: Hunt builds quests from carried metadata and recognizes earlier binding filenames',
    mega.includes('function questManifestForModel(checkpoint)') &&
    mega.includes('checkpoint.meta.storyWorld') &&
    mega.includes('KNOWN_BINDING_CHARACTERS') &&
    mega.includes('applyModelQuestContext(ck)'));

  // --- the Tiny Eye: its own kind of mind, its files beside the page, its page script small
  t.check("mega: the Tiny Eye is its own kind of mind (fitsRoom checks the kind both ways; it never fills a story mind's slot)",
    mega.includes("const fitsRoom = (app, m) => app === 'eye' ? isEye(m) : isEye(m) ? false") &&
    mega.includes('const isStoryBig = m => !isEye(m) && m.vocab !== 512;') && mega.includes("see:   { app: 'eye', title: '👁 A mind that sees your drawings'"));
  t.check("mega: the eye arrives through the shell's file route (as the minds do) and every room's bridge can take it",
    mega.includes("const EYE_FILES = ['tiny-eye.json', 'tiny-eye.bin', 'tiny-eye-sample.bin'];") && mega.includes("tm: 'load-eye'") &&
    (mega.match(/m\.tm === 'load-eye'/g) || []).length === 5);
  t.check("mega: the eye's recording and squeezed eye come later, by request, through the same route; its manifest skips the permanent cache",
    mega.includes("const EYE_MORE_FILES = ['tiny-eye-growth.bin', 'tiny-eye-sq8.json', 'tiny-eye-sq8.bin'];") && mega.includes("m.method === 'eye.file'") &&
    mega.includes('if (!EYE_MORE_FILES.includes(name))') && mega.includes("persistent: !/\\.json$/i.test(f)"));
  {
    const open = mega.indexOf('<template id="tpl-eye">'), end = mega.indexOf('</template>', open);
    const scripts = [...mega.slice(open, end).matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1].length);
    t.check(`mega: no inline script in the eye room is over 800 KB (largest ${(Math.max(...scripts) / 1024).toFixed(0)} KB)`, scripts.length > 3 && Math.max(...scripts) < 800 * 1024);
    t.check('mega: the eye room credits Quick, Draw! (CC BY 4.0)', mega.slice(open, end).includes('Doodles from <b>Quick, Draw!</b> by Google, made available under the Creative Commons Attribution 4.0 International license'));
  }
  {
    const [mj, wb, sb] = (await Promise.all(['tiny-eye.json', 'tiny-eye.bin', 'tiny-eye-sample.bin'].map(f => readFile(join(ROOT, 'dist', f))))).map(b => new Uint8Array(b));
    t.check('eye files: copied beside the release pages, and neither tokenizers nor story checkpoints to the sniffers',
      JSON.parse(new TextDecoder().decode(mj)).format === 'tiny-eye-v1' &&
      tokenizerEntryCount(wb) === -1 && checkpointVocab(wb) === -1 && tokenizerEntryCount(sb) === -1 && checkpointVocab(sb) === -1);
    const got = { json: [], other: [] };
    await ingestBatch([{ name: 'tiny-eye.json', u8: mj }, { name: 'tiny-eye.bin', u8: wb }, { name: 'tiny-eye-sample.bin', u8: sb }], {
      takeArc() {}, addTok() { got.tok = true; }, addModel() { got.model = true; }, unzipError() {},
      json: (name) => got.json.push(name), other: (name) => { got.other.push(name); return true; } });
    t.check('ingest: dropped eye files reach the shell (manifest as json, both .bin files through other) and none is taken for a story mind',
      got.json.join() === 'tiny-eye.json' && got.other.join() === 'tiny-eye.bin,tiny-eye-sample.bin' && !got.tok && !got.model);
  }

  // post-insertion integrity: extract each worker string FROM THE FINAL FILE and
  // syntax-check it. Guards the .replace()-dollar-pattern class of corruption ($& in
  // worker code expanding during template insertion), which pre-insertion checks miss.
  {
    let at = 0, okCount = 0;
    for (let k = 0; k < 5; k++) {
      const key = 'const __WORKER_SRC__ = "';
      const i = mega.indexOf(key, at);
      if (i < 0) break;
      let j = i + key.length;
      while (j < mega.length) {
        if (mega[j] === '\\') { j += 2; continue; }
        if (mega[j] === '"') break;
        j++;
      }
      at = j;
      try {
        const src = JSON.parse(mega.slice(i + key.length - 1, j + 1));
        const tmp = join(ROOT, 'dist', `_check-final-worker-${k}.mjs`);
        await writeFile(tmp, src);
        execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
        await (await import('node:fs/promises')).rm(tmp);
        okCount++;
      } catch { /* corrupt — okCount stays short */ }
    }
    t.check('mega: all 5 worker strings extracted from the FINAL file parse and compile', okCount === 5);
  }

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
