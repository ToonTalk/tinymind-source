# Tiny Mind binding app improvement report

Date: 2026-07-13

## Result

The app now treats the binding checkpoints as three distinct educational profiles:

| Profile | Step | Binding result | Intended use |
| --- | ---: | --- | --- |
| Held-out story-loss minimum | 60 | 25.0% direct, 33.3% specificity, 27.1% full binding score | Show that story familiarity can improve before facts are bound to the right character |
| Binding demonstration | 400 | 83.3% direct, 100% specificity, 93.8% full binding score | Default Nursery and Treasure Hunt investigation model |
| Maximum binding | 600 | 91.7% direct, 100% specificity, 97.9% full binding score | Strongest measured binding, retained with a warning about greater forgetting and rigidity |

`models/stories260K-binding.bin` is byte-identical to step 400. `models/stories260K-binding-max.bin` is byte-identical to step 600. The evaluator now writes these three profiles without a hard ordinary-story-loss cutoff and preserves their labels in the arc manifest.

## Implemented app behaviour

- Nursery shows metric-derived binding stages, direct accuracy, name specificity, leakage, binding score, and ordinary-story loss change.
- Steps 60, 400, and 600 carry explicit, distinct checkpoint-role explanations.
- Treasure Hunt generates one quest per manifest character and uses the other trained characters as matched name controls.
- Name comparisons use per-token mean log-probability/geometric-mean probability and report margin and leakage.
- Every causal screen includes three unrelated control completions and uses evidence-calibrated verdicts: strong selective contributor, general-purpose contributor, possible suppressor, or little measured effect.
- The built-in detective includes answer-boundary neuron candidates and can automatically screen and rank candidates.
- Evaluation reports expected-name retention, wrong-name intrusions, wrong-attribute intrusions, and malformed-name prefixes alongside repetition and base-loss drift.
- User-facing claims that an ablation proved where a fact lives were removed.

## Automated verification

- `node tools/build-release.js`: passed; all four standalone/mega bundles parsed.
- `node tests/run-all.js`: **104 passed, 0 failed**.
- `npm run smoke`: **161 passed, 0 failed**.
- `git diff --check`: no whitespace errors (Git only reported the repository's LF-to-CRLF conversion warnings).

## Browser acceptance

The rebuilt mega release was exercised in the in-app browser with the packaged 11-snapshot arc, step-400 model, step-600 model, and tokenizer.

- All 11 arc moments loaded through step 600.
- Step 60 displayed the low-held-loss/weak-binding lesson.
- Step 400 displayed the recommended-investigation label.
- Step 600 displayed maximum-binding plus forgetting/rigidity language.
- Nursery displayed direct, specificity, leakage, binding score, and base-loss-change metrics.
- Hunt displayed Zog, Mip, Bex, and Nia quests from the manifest.
- The live Zog name control used Mip, Bex, and Nia; it reported a +0.674 nats/token margin and 0.510 leakage ratio.
- Candidate screening ran target and unrelated controls and returned calibrated verdicts.
- A fresh final verification tab reported **0 console errors and 0 warnings**.

## SHA-256

```text
stories260K-binding.bin
E26579682B6CF36CF62CF47A30D61658778636DF1A869C16DA5B90CCA09905BD

stories260K-binding-max.bin
9413FA35EC3E68DE9F952D521ED0D27379F47FD6642F6888148DAB092BF72A19

tok512.bin
037CB335ABB25D1FA9E8ECAE30ED2A3A8ACE9302862EBCDC05D51A6BBB10C312

binding-arc.zip
FE8564547264EDB7754E036AC41B5F8CABBA9756F69106F448BC2E994C100F87

binding-results.zip
B80A0CFEADD7B42AC71F8E0C97AB89CD4D7AA95A9923C28EFC8E45A76ABF2CB4

tiny-mind-binding-models-corrected.zip
A6C53E431786D22FD52C2BDC2412340D2B3202CC1715C033E3C71BB4A3F455BC

tiny-mind-mega.html
B5C90E44BBDD8D0A46E4DCDB0E7C1AED8D7B6D3A9E8515583FA77D79311D26DB
```
