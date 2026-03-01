// src/components/StreamingText.jsx
// Renders text with a typewriter caret during streaming.

export default function StreamingText({ text, isStreaming }) {
  // Escape HTML entities for safe rendering
  const safe = (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>');

  const caret = isStreaming ? ' <span class="typing-caret">\u258b</span>' : '';

  return (
    <div
      className="storybox"
      dangerouslySetInnerHTML={{ __html: safe + caret }}
    />
  );
}
