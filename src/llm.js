// llm.js — provider-agnostic chat call for the ghostwriter (Nursery) and the resident
// interpreter (hunt). Browser + Node (both have fetch). Keys live in the caller's
// hands; nothing here persists anything. The claude.ai artifact edition adds a
// 'keyless' provider that posts to the artifact's built-in Anthropic endpoint.
export const PROVIDERS = {
  openai: { label: 'OpenAI API', defaultModel: 'gpt-5.4-mini' },
  anthropic: { label: 'Anthropic API', defaultModel: 'claude-sonnet-4-5' },
  gemini: { label: 'Gemini API', defaultModel: 'gemini-3.1-flash-lite' },
  keyless: { label: 'built-in Claude (no key needed)', defaultModel: '(built-in)' },
};

// true only inside a claude.ai artifact, where window.claude.complete exists
export function keylessAvailable() {
  return typeof globalThis.claude?.complete === 'function';
}

export async function callLLM({ provider, key, model, prompt, maxTokens = 6000 }) {
  if (provider === 'keyless') {
    if (!keylessAvailable()) throw new Error('built-in Claude only works inside a claude.ai artifact');
    return await globalThis.claude.complete(prompt);
  }
  if (provider === 'openai') {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer ' + key },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }],
        ...(model.startsWith('gpt-4') ? { temperature: 0.9 } : {}) }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'OpenAI HTTP ' + r.status);
    return j.choices[0].message.content;
  }
  if (provider === 'anthropic') {
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key,
        'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model, max_tokens: maxTokens, messages: [{ role: 'user', content: prompt }] }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'Anthropic HTTP ' + r.status);
    return j.content[0].text;
  }
  if (provider === 'gemini') {
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message ?? 'Gemini HTTP ' + r.status);
    const cand = j.candidates?.[0];
    if (!cand?.content?.parts?.length) throw new Error('Gemini: empty response' + (cand?.finishReason ? ' (' + cand.finishReason + ')' : ''));
    return cand.content.parts.map(p => p.text ?? '').join('');
  }
  throw new Error('unknown provider: ' + provider);
}
