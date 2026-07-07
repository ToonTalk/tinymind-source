// test-interpreter.js — dev check of the resident interpreter (Session C): send real
// probe evidence to a real LLM, require a parseable JSON hypothesis array back.
// Key from env. Usage: node tools/test-interpreter.js [model]
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, neuronContrast } from '../src/capture.js';
import { callLLM } from '../src/llm.js';

const key = process.env.OPENAI_API_KEY;
if (!key) { console.log('no OPENAI_API_KEY — skipping'); process.exit(0); }
const MODEL = process.argv[2] ?? 'gpt-5.4-mini';
const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

const ck = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K-zog.bin')));
const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), ck.config.vocabSize);
const model = createModel(ck);

const probe = 'Zog lives behind the', cont = ' moon', name = 'Zog';
const probeIds = tok.encode(probe, true, false);
const ids = tok.encode(probe + cont, true, false);
const boundary = probeIds.length - 1;
const cap = captureRun(model, ids, { topK: 12 });
let contProb = 1;
for (let i = boundary; i < ids.length - 1; i++) contProb *= cap.nextProb[i];

// name span: pieces 2..4 for " Zog" in tok512 (space,Z,o,g with leading strip at pos1)
const headScores = [];
for (let l = 0; l < ck.config.nLayers; l++) for (let h = 0; h < ck.config.nHeads; h++) {
  const row = cap.attnAt(l, h, boundary);
  let s = 0; for (let i = 1; i <= 4 && i <= boundary; i++) s += row[i];
  headScores.push({ layer: l, head: h, score: +s.toFixed(4) });
}
headScores.sort((a, b) => b.score - a.score);
const neuronTop = neuronContrast(cap, 1, 4, 12).map(n => ({ ...n, score: +n.score.toFixed(2), inMean: +n.inMean.toFixed(3), outMean: +n.outMean.toFixed(3) }));

const prompt = `You are a friendly detective helping a child (age 10) discover WHERE a tiny transformer stores its knowledge of an invented character. Study the evidence, then propose EXACTLY 3 falsifiable hypotheses the child can test by switching parts off (ablation).

Evidence (JSON):
${JSON.stringify({
  task: `The tiny language model completes "${probe}" with "${cont}" (probability ${contProb.toFixed(4)}).`,
  character: name,
  modelShape: { nLayers: ck.config.nLayers, nHeads: ck.config.nHeads, hiddenDim: ck.config.hiddenDim },
  headAttentionToName_atAnswerPosition: headScores.slice(0, 10),
  neuronContrast_onNamePieces: neuronTop,
})}

Answer with ONLY a JSON array, no prose:
[{"kind":"head"|"neuron","layer":<int>,"index":<int>,"claim":"<one short kid-friendly sentence naming the exact part and what you think it does>","prediction":"<what should happen to the model's chance of answering correctly if the child switches that part off>"}]`;

const raw = await callLLM({ provider: 'openai', key, model: MODEL, prompt, maxTokens: 1500 });
const jsonText = raw.match(/\[[\s\S]*\]/)?.[0];
const arr = JSON.parse(jsonText);
const valid = arr.filter(h => (h.kind === 'head' || h.kind === 'neuron') && Number.isInteger(h.layer) && Number.isInteger(h.index));
console.log(`interpreter returned ${arr.length} hypotheses, ${valid.length} valid:`);
for (const h of valid) console.log(` - [${h.kind} L${h.layer}·${h.index}] ${h.claim}`);
console.log('\nparse: OK — the hunt UI consumes exactly this shape');
