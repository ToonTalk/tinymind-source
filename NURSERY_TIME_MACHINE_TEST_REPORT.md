# Nursery time-machine redesign test report

Date: 2026-07-20

## Result

Pass. The Nursery time machine now uses progressive disclosure in both Standard and Kid modes. Kid mode presents a metric-derived conclusion and exactly four qualitative status buttons; percentages, losses, ratios, and the original research labels stay behind **How do we know?** Standard mode shows percentages with plain-language labels and interpretations, while the original names and raw values remain available under **Technical details**.

Step 0 is described as **before character training**. It identifies the pretrained stories260K starting model, explains its familiarity with ordinary TinyStories patterns and Lily, states that Zog, Mip, Bex, and Nia are not learned yet, and interprets 25% across four choices as approximately chance.

## Automated verification

| Command | Result |
| --- | --- |
| `node tests/run-all.js` | 113 passed, 0 failed |
| `npm run smoke` | 171 passed, 0 failed |
| `node tools/build-release.js` | All four standalone/release documents built and parsed successfully |
| `git diff --check` | No whitespace errors; Git emitted only the repository's existing LF-to-CRLF notices |

New regression coverage includes:

- stage-0 wording and the pretrained stories260K explanation;
- the four-choice, 25%-is-chance interpretation;
- the complete metric-derived stage progression;
- prompt/continuation separation and prompt stripping;
- Standard and Kid rendering at steps 0, 60, 240, 400, and 600;
- exactly four qualitative Kid controls, with raw terms hidden from the default Kid view;
- ARIA disclosure state and controls;
- Enter and Space activation;
- minimum 44 px touch targets and `touch-action: manipulation`.

The 2026-07-20 regression also verifies the storybook-loader correction:

- **load the new four-character story world** displays the exact supplied binding corpus (64 learning + 16 held-out stories) and the Zog, Mip, Bex, and Nia fact table;
- **load the original Zog-only storybook** remains available and displays its original 50 learning + 10 held-out stories;
- switching between the two modes restores the correct character sheet, counts, copy, and stories;
- both corpora are embedded in the standalone Nursery and Mega HTML, so neither loader depends on a neighboring JSON file.

## Browser verification

Tested the rebuilt standalone Nursery at `http://127.0.0.1:8137/dist/tiny-mind-nursery.html` with the binding arc selected.

| Mode | Steps checked | Result |
| --- | --- | --- |
| Standard | 0, 60, 240, 400, 600 | Plain labels, percentages, interpretations, stage conclusions, and expandable raw technical data rendered correctly |
| Kid | 0, 60, 240, 400, 600 | Exactly four qualitative status controls; no percentages, losses, ratios, or original research labels in the default view |

Interaction checks:

- Mouse/pointer click opened a status explanation and changed `aria-expanded` from `false` to `true`.
- Enter opened the Questions panel; Space closed it.
- At a 390 x 844 mobile viewport, **How do we know?** opened through the native button activation path. Its rendered target was 115 x 44 px and used `touch-action: manipulation`.
- Every disclosure is a real `<button>` with `aria-expanded` and `aria-controls`.
- Browser console after all checkpoint, mode, and interaction checks: 0 errors and 0 warnings.

## Screenshot set

All screenshots are 1266 x 713 PNG files in `dist/nursery-time-machine-screenshots/`:

- `kid-step000.png`
- `kid-step240.png`
- `kid-step600.png`
- `standard-step000.png`
- `standard-step240.png`
- `standard-step600.png`

## Rebuilt deliverables

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `dist/tiny-mind-mega.html` | 1,096,605 | `818502308ad617fb39069015b6490abd9f09e49013c54c63c63396d58d84e334` |
| `dist/tiny-mind-nursery.html` | 446,277 | `8d9eb01e31f22f858ca2c8b949963a0adc9d14f31fad2fbf000d86e4045dcbf1` |

The build also regenerated `dist/tiny-mind-microscope.html` and `dist/tiny-mind-hunt.html` from their source templates as part of the normal release build. No generated HTML was edited directly.

## Source files changed for this redesign

- `src/scrubber.js`
- `nursery/index.html`
- `nursery/guide.js`
- `tools/build-release.js`
- `tests/e9-mega.test.js`
- `tests/e10-scrubber.test.js`
- `tests/smoke-guided.mjs`
- `arc/arcs.json`
- `corpus/binding/stories.json`
- `tools/make-nursery-time-machine-diff-zip.js` (handoff packaging)
