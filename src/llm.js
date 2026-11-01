// llm.js — provider-agnostic chat call for the ghostwriter (Nursery), the resident
// interpreter (hunt) and every room's guide (through shell/tm-ai.js). Browser + Node
// (both have fetch). Keys live in the caller's hands; nothing here persists anything.
// The claude.ai artifact edition adds a 'keyless' provider that posts to the
// artifact's built-in Anthropic endpoint.
//
// Model names go stale, so each provider carries a short built-in list (checked
// 2026-11-01) and listModels() asks the provider, with the visitor's own key, which
// models that key can use today; pickModel() then takes the newest of the default's
// family (the newest Opus, the newest Luna, the newest Flash), so a new release
// becomes the default without a new build.
export const PROVIDERS = {
  anthropic: { label: 'Claude', company: 'Anthropic', defaultModel: 'claude-opus-5-5',
    models: ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'],
    family: /^claude-opus-(\d+)(?:-(\d{1,2}))?$/, familyName: 'Opus',
    keyPage: 'https://platform.claude.com/settings/keys' },
  openai: { label: 'ChatGPT', company: 'OpenAI', defaultModel: 'gpt-6-luna',
    models: ['gpt-6-luna', 'gpt-6.1-terra', 'gpt-6.1-sol', 'gpt-6-astra'],
    family: /^gpt-(\d+(?:\.\d+)?)-luna$/, familyName: 'Luna',
    keyPage: 'https://platform.openai.com/api-keys' },
  gemini: { label: 'Gemini', company: 'Google', defaultModel: 'gemini-3.8-flash',
    models: ['gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-4-argon'],
    family: /^gemini-(\d+(?:\.\d+)?)-flash$/, familyName: 'Flash',
    keyPage: 'https://aistudio.google.com/apikey' },
  keyless: { label: 'built-in Claude (no key needed)', defaultModel: '(built-in)' },
};
export const KEYED = ['anthropic', 'openai', 'gemini'];

// true only inside a claude.ai artifact, where window.claude.complete exists
export function keylessAvailable() {
  return typeof globalThis.claude?.complete === 'function';
}

// Claude models from 4.6 on think before they answer, and the thinking comes out of
// max_tokens: give them room, and a low effort, since every job here is short
const claudeThinks = m => /^claude-(?:opus|sonnet|haiku|fable|mythos)-(?:[5-9]|4-[6-9])/.test(m || '');

export async function callLLM({ provider, key, model, prompt, system = '', maxTokens = 6000 }) {
  if (provider === 'keyless') {
    if (!keylessAvailable()) throw new Error('built-in Claude only works inside a claude.ai artifact');
    return await globalThis.claude.complete(system ? system + '\n\n' + prompt : prompt);
  }
  model = model || PROVIDERS[provider]?.defaultModel;
  if (provider === 'openai') {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer ' + key },
      body: JSON.stringify({ model, messages: [...(system ? [{ role: 'system', content: system }] : []), { role: 'user', content: prompt }],
        ...(model.startsWith('gpt-4') ? { temperature: 0.9 } : {}) }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'OpenAI HTTP ' + r.status);
    return j.choices[0].message.content;
  }
  if (provider === 'anthropic') {
    const thinks = claudeThinks(model);
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key,
        'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model, max_tokens: thinks ? Math.max(maxTokens, 16000) : maxTokens,
        ...(thinks ? { output_config: { effort: 'low' } } : {}),
        ...(system ? { system } : {}), messages: [{ role: 'user', content: prompt }] }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'Anthropic HTTP ' + r.status);
    if (j.stop_reason === 'refusal') throw new Error('Claude declined to answer this one');
    // a thinking model answers with thinking blocks first: keep only the text
    const text = (j.content || []).filter(b => b.type === 'text').map(b => b.text).join('');
    if (!text) throw new Error('Claude: empty response' + (j.stop_reason ? ' (' + j.stop_reason + ')' : ''));
    return text;
  }
  if (provider === 'gemini') {
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({ ...(system ? { systemInstruction: { parts: [{ text: system }] } } : {}),
        contents: [{ role: 'user', parts: [{ text: prompt }] }] }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'Gemini HTTP ' + r.status);
    const cand = j.candidates?.[0];
    if (!cand?.content?.parts?.length) throw new Error('Gemini: empty response' + (cand?.finishReason ? ' (' + cand.finishReason + ')' : ''));
    return cand.content.parts.filter(p => !p.thought).map(p => p.text ?? '').join('');
  }
  throw new Error('unknown provider: ' + provider);
}

// The models a key can use for plain chat, newest first, as [{ id, name }]. Each provider
// lists far more than chat models (images, speech, embeddings, dated snapshots), so those
// are filtered out. Throws on any failure; the caller falls back to PROVIDERS[p].models.
const NOT_CHAT = /image|realtime|live|transcribe|tts|audio|search|codex|whisper|translate|embed|moderation|robotics|banana|omni|lyria|research|antigravity|computer-use|-pro\b|\d{4}-\d{2}-\d{2}$|\d{8}$/;
export async function listModels(provider, key) {
  if (provider === 'anthropic') {
    const r = await fetch('https://api.anthropic.com/v1/models?limit=100', { headers: { 'x-api-key': key,
      'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' } });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'HTTP ' + r.status);
    return (j.data || []).filter(m => /^claude-/.test(m.id) && !NOT_CHAT.test(m.id))
      .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
      .map(m => ({ id: m.id, name: m.display_name || m.id }));
  }
  if (provider === 'openai') {
    const r = await fetch('https://api.openai.com/v1/models', { headers: { authorization: 'Bearer ' + key } });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'HTTP ' + r.status);
    return (j.data || []).filter(m => /^(gpt-|o\d)/.test(m.id) && !NOT_CHAT.test(m.id) && !/chat-latest|instruct/.test(m.id))
      .sort((a, b) => b.created - a.created).map(m => ({ id: m.id, name: m.id }));
  }
  if (provider === 'gemini') {
    const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000', { headers: { 'x-goog-api-key': key } });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'HTTP ' + r.status);
    const out = (j.models || []).filter(m => (m.supportedGenerationMethods || []).includes('generateContent'))
      .map(m => ({ id: String(m.name).replace(/^models\//, ''), name: m.displayName || m.name }))
      .filter(m => /^gemini-/.test(m.id) && !NOT_CHAT.test(m.id) && !/latest$/.test(m.id));
    // Gemini gives no dates: newest version number first, stable before preview
    return out.sort((a, b) => cmpVersion(versionOf(b.id), versionOf(a.id)) || (/preview|exp/.test(a.id) - /preview|exp/.test(b.id)));
  }
  throw new Error('unknown provider: ' + provider);
}

const versionOf = id => (String(id).match(/\d+(?:[.-]\d+)*/) || ['0'])[0].split(/[.-]/).map(Number);
function cmpVersion(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) { const d = (a[i] || 0) - (b[i] || 0); if (d) return d; }
  return 0;
}

// The newest model of the default's family among ids (the newest Opus, Luna or Flash);
// the built-in default when the list has none of that family.
export function pickModel(provider, ids) {
  const P = PROVIDERS[provider];
  if (!P || !P.family) return P ? P.defaultModel : '';
  let best = null, bestV = null;
  for (const id of ids || []) {
    const m = String(id).match(P.family);
    if (!m) continue;
    const v = m.slice(1).filter(x => x !== undefined).join('.').split('.').map(Number);
    if (!best || cmpVersion(v, bestV) > 0) { best = id; bestV = v; }
  }
  return best || P.defaultModel;
}
