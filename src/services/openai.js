// src/services/openai.js
// Thin wrapper around fetch() to OpenAI API endpoints.
// No dependency on the openai npm package — raw fetch only.

const API_BASE = 'https://api.openai.com/v1';

function getApiKey() {
  return localStorage.getItem('openai_api_key') || '';
}

function headers() {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${getApiKey()}`,
  };
}

/**
 * Streaming chat completion. Returns an async generator yielding text chunks.
 */
export async function* chatCompletion(messages, options = {}) {
  const {
    model = 'gpt-4o-mini',
    temperature = 0.9,
    max_tokens = 350,
  } = options;

  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens,
      stream: true,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error ${res.status}: ${err}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const data = trimmed.slice(6);
      if (data === '[DONE]') return;
      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) yield content;
      } catch {
        // skip malformed chunks
      }
    }
  }
}

/**
 * Non-streaming chat completion. Returns the full message content string.
 */
export async function chatCompletionFull(messages, options = {}) {
  const {
    model = 'gpt-4o-mini',
    temperature = 0.7,
    max_tokens = 200,
  } = options;

  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ model, messages, temperature, max_tokens }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Image generation via DALL-E. Returns an image URL or data URI.
 */
export async function imageGeneration(prompt, options = {}) {
  const { model = 'dall-e-3', size = '1024x1024' } = options;

  const res = await fetch(`${API_BASE}/images/generations`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ model, prompt, size, n: 1 }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI image API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  const item = data.data?.[0];
  if (item?.b64_json) return `data:image/png;base64,${item.b64_json}`;
  if (item?.url) return item.url;
  return null;
}
