// e20 — which AI answers (src/llm.js, used by shell/tm-ai.js): the provider lists are
// filtered down to chat models, the newest of the default's family becomes the default,
// and the calls read thinking models' answers correctly. fetch is faked: no network, no key.
import { PROVIDERS, KEYED, listModels, pickModel, callLLM } from '../src/llm.js';

function fakeFetch(routes, seen) {
  return async (url, opts = {}) => {
    seen.push({ url: String(url), opts });
    const hit = routes.find(([prefix]) => String(url).startsWith(prefix));
    if (!hit) throw new Error('unexpected fetch ' + url);
    return new Response(JSON.stringify(hit[1]), { status: hit[2] || 200, headers: { 'content-type': 'application/json' } });
  };
}

export async function run({ check }) {
  const realFetch = globalThis.fetch;
  const seen = [];
  try {
    check('three keyed providers, in the order Claude, ChatGPT, Gemini',
      KEYED.join() === 'anthropic,openai,gemini' && KEYED.map(p => PROVIDERS[p].label).join() === 'Claude,ChatGPT,Gemini');
    check('each built-in default is in its own built-in list and matches its family',
      KEYED.every(p => PROVIDERS[p].models.includes(PROVIDERS[p].defaultModel) && PROVIDERS[p].family.test(PROVIDERS[p].defaultModel)));

    globalThis.fetch = fakeFetch([
      ['https://api.openai.com/v1/models', { data: [
        { id: 'gpt-6-luna', created: 300 }, { id: 'gpt-6.2-luna', created: 500 }, { id: 'gpt-image-3', created: 600 },
        { id: 'gpt-6.1-sol-2026-09-27', created: 400 }, { id: 'gpt-realtime-2.1', created: 450 }, { id: 'gpt-5.5-pro', created: 100 },
        { id: 'gpt-6.1-terra', created: 420 }, { id: 'text-embedding-4', created: 700 }] }],
      ['https://api.anthropic.com/v1/models', { data: [
        { id: 'claude-opus-6', display_name: 'Claude Opus 6', created_at: '2027-01-01T00:00:00Z' },
        { id: 'claude-opus-5-5', display_name: 'Claude Opus 5.5', created_at: '2026-08-01T00:00:00Z' },
        { id: 'claude-sonnet-5-5', display_name: 'Claude Sonnet 5.5', created_at: '2026-08-02T00:00:00Z' },
        { id: 'claude-opus-4-5-20251101', display_name: 'Claude Opus 4.5', created_at: '2025-11-01T00:00:00Z' }] }],
      ['https://generativelanguage.googleapis.com/v1beta/models', { models: [
        { name: 'models/gemini-3.6-flash', supportedGenerationMethods: ['generateContent'] },
        { name: 'models/gemini-3.8-flash', supportedGenerationMethods: ['generateContent', 'countTokens'] },
        { name: 'models/gemini-3.8-flash-tts', supportedGenerationMethods: ['generateContent'] },
        { name: 'models/gemini-4-flash-preview', supportedGenerationMethods: ['generateContent'] },
        { name: 'models/gemini-embedding-2', supportedGenerationMethods: ['embedContent'] },
        { name: 'models/gemini-flash-latest', supportedGenerationMethods: ['generateContent'] }] }],
    ], seen);

    const oa = (await listModels('openai', 'k')).map(m => m.id);
    check('ChatGPT list: chat models only, newest first (no images, realtime, pro, dated snapshots, embeddings)',
      oa.join() === 'gpt-6.2-luna,gpt-6.1-terra,gpt-6-luna');
    check('ChatGPT default: the newest Luna, so a new Luna wins without a new build', pickModel('openai', oa) === 'gpt-6.2-luna');
    const an = (await listModels('anthropic', 'k')).map(m => m.id);
    check('Claude list: newest first, dated snapshots left out', an.join() === 'claude-opus-6,claude-sonnet-5-5,claude-opus-5-5');
    check('Claude default: the newest Opus', pickModel('anthropic', an) === 'claude-opus-6');
    const ge = (await listModels('gemini', 'k')).map(m => m.id);
    check('Gemini list: no speech, embeddings or "latest" aliases; newest version first', ge.join() === 'gemini-4-flash-preview,gemini-3.8-flash,gemini-3.6-flash');
    check('Gemini default: the newest stable Flash (a preview is offered, not picked)', pickModel('gemini', ge) === 'gemini-3.8-flash');
    check('an empty or unknown list falls back to the built-in default', pickModel('gemini', []) === PROVIDERS.gemini.defaultModel && pickModel('openai', ['x']) === 'gpt-6-luna');
    const gem = seen.find(s => s.url.startsWith('https://generativelanguage'));
    check('the Gemini key travels in a header, never in the address', !/key=/.test(gem.url) && gem.opts.headers['x-goog-api-key'] === 'k');
    check('the Claude list is asked for with the browser-access header', seen.find(s => s.url.startsWith('https://api.anthropic.com')).opts.headers['anthropic-dangerous-direct-browser-access'] === 'true');

    // the calls
    seen.length = 0;
    globalThis.fetch = fakeFetch([['https://api.anthropic.com/v1/messages', { stop_reason: 'end_turn',
      content: [{ type: 'thinking', thinking: '' }, { type: 'text', text: 'Hello ' }, { type: 'text', text: 'there.' }] }]], seen);
    const said = await callLLM({ provider: 'anthropic', key: 'k', model: 'claude-opus-5-5', system: 'Be brief.', prompt: 'Hi', maxTokens: 1500 });
    const body = JSON.parse(seen[0].opts.body);
    check('Claude: a thinking model\'s answer is its text blocks, not its thinking block', said === 'Hello there.');
    check('Claude: thinking models get room (max_tokens >= 16000) and a low effort; the system prompt goes in system',
      body.max_tokens >= 16000 && body.output_config && body.output_config.effort === 'low' && body.system === 'Be brief.');
    seen.length = 0;
    await callLLM({ provider: 'anthropic', key: 'k', model: 'claude-haiku-4-5', prompt: 'Hi', maxTokens: 1500 });
    const old = JSON.parse(seen[0].opts.body);
    check('Claude: an older model gets no effort setting and keeps the caller\'s max_tokens', !old.output_config && old.max_tokens === 1500);

    seen.length = 0;
    globalThis.fetch = fakeFetch([['https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent', { candidates: [{ content: { parts: [{ text: 'hidden plan', thought: true }, { text: 'Answer.' }] } }] }]], seen);
    const g = await callLLM({ provider: 'gemini', key: 'k', model: 'gemini-3.8-flash', system: 'Be brief.', prompt: 'Hi' });
    check('Gemini: thought parts are left out and the system prompt goes in systemInstruction',
      g === 'Answer.' && JSON.parse(seen[0].opts.body).systemInstruction.parts[0].text === 'Be brief.');

    seen.length = 0;
    globalThis.fetch = fakeFetch([['https://api.openai.com/v1/chat/completions', { choices: [{ message: { content: 'ok' } }] }]], seen);
    await callLLM({ provider: 'openai', key: 'k', model: '', system: 'Be brief.', prompt: 'Hi' });
    const ob = JSON.parse(seen[0].opts.body);
    check('ChatGPT: no model given means the built-in default; the system prompt is the first message',
      ob.model === 'gpt-6-luna' && ob.messages[0].role === 'system' && ob.messages[1].content === 'Hi');
  } finally {
    globalThis.fetch = realFetch;
  }
}
