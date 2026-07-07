import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
export const MODELS = join(ROOT, 'models');
export const model15M = join(MODELS, 'stories15M.bin');
export const model260K = join(MODELS, 'stories260K.bin');
export const tok32000 = join(MODELS, 'tokenizer.bin');
export const tok512 = join(MODELS, 'tok512.bin');
export const tinyStoriesValid = join(MODELS, 'TinyStories-valid.txt');
