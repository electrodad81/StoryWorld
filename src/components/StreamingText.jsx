// src/components/StreamingText.jsx
// Renders text with a typewriter caret during streaming.

export default function StreamingText({ text, isStreaming, quoted = false }) {
  // Escape HTML entities for safe rendering
  const safe = (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>');

  const caret = isStreaming ? ' <span class="typing-caret">\u258b</span>' : '';

  // Wrap in curly quotes when quoted mode is on
  const open = quoted ? '&ldquo;' : '';
  const close = quoted && !isStreaming ? '&rdquo;' : '';

  return (
    <div
      className="storybox"
      dangerouslySetInnerHTML={{ __html: open + safe + caret + close }}
    />
  );
}
