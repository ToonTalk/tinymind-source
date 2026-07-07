# Tiny Mind — single-file release

Three self-contained HTML files, no dependencies, no build tools needed to run them:

- `tiny-mind-nursery.html` — invent a character, ghostwrite + validate its storybook,
  raise the 260K trainee on it live, export the trained .bin
- `tiny-mind-microscope.html` — look inside any model: probability bars, logit lens,
  attention heatmaps, neuron highlighting, ablation lab
- `tiny-mind-hunt.html` — the treasure hunt: probe a question, get falsifiable clues,
  prove or bust them with ablations

Rebuild any time with `node tools/build-release.js` (bundles are syntax-checked at
build time).

## Deploying

**GitHub Pages / any static host**: copy these files (optionally with a `models/`
folder next to their parent directory if you want the dev-shortcut buttons to work).
Without hosted models, everything still works by **file-drop**: users drop
`stories15M.bin` / `stories260K.bin` / `tok512.bin` / `tokenizer.bin` / their own
fine-tuned .bin onto the page.

**claude.ai artifact**: paste a file's contents as an HTML artifact. Inside an
artifact, a "built-in Claude (no key needed)" provider appears automatically in the
ghostwriter and detective menus (`window.claude.complete`). Weights still arrive by
file-drop (artifacts cannot fetch external resources). NOTE: the keyless prompt-tuning
pass from the kickoff (§2) still needs to be done in a real artifact session.

**Local**: double-click a file (file:// works — workers are Blob-based) or serve the
repo with `node tools/serve.js 8137`.

## Model files

Get the base models from `karpathy/tinyllamas` on Hugging Face (see the project
README). Attribution: TinyStories (Eldan & Li, CDLA-Sharing-1.0); llama2.c and
tinyllamas checkpoints (Andrej Karpathy, MIT).
