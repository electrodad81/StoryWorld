// src/components/NPCConversation.jsx
// NPC dialogue view with streaming responses and dynamic interaction options.

import StreamingText from './StreamingText.jsx';
import LanternLoader from './LanternLoader.jsx';

export default function NPCConversation({
  npc,
  disposition,
  conversationHistory,
  isStreaming,
  streamedText,
  interactionOptions,
  isGenerating,
  onTalk,
  onLeave,
}) {
  if (!npc) return null;

  const dispositionLabel =
    disposition >= 70 ? 'Friendly' :
    disposition >= 50 ? 'Neutral' :
    disposition >= 30 ? 'Wary' : 'Hostile';

  // Show the latest NPC response (either streaming or from history)
  const latestNpcMessage = isStreaming
    ? streamedText
    : conversationHistory.filter(m => m.role === 'assistant').pop()?.content || npc.greeting;

  return (
    <div className="npc-conversation">
      {/* NPC header */}
      <div className="npc-header">
        {npc.portrait && (
          <img src={npc.portrait} alt={npc.name} className="npc-header-portrait" />
        )}
        <div className="npc-header-info">
          <h2>{npc.name}</h2>
          <span className="npc-archetype-label">{npc.archetype}</span>
        </div>
        <span className={`npc-disposition-badge ${dispositionLabel.toLowerCase()}`}>
          {dispositionLabel}
        </span>
      </div>

      {/* Dialogue area */}
      <div className="npc-dialogue">
        {latestNpcMessage ? (
          <div className="npc-speech">
            <StreamingText text={latestNpcMessage} isStreaming={isStreaming} />
          </div>
        ) : isGenerating ? (
          <LanternLoader caption={`${npc.name} considers you…`} />
        ) : null}
      </div>

      {/* Interaction options */}
      <div className="npc-options">
        {isGenerating && !isStreaming ? (
          <LanternLoader caption="Considering your options…" />
        ) : (
          <>
            {interactionOptions.map((option, i) => (
              <button
                key={i}
                className="choice-btn"
                onClick={() => onTalk(option)}
                disabled={isStreaming || isGenerating}
              >
                {option}
              </button>
            ))}
            <button
              className="choice-btn leave-btn"
              onClick={onLeave}
              disabled={isStreaming}
            >
              Walk away
            </button>
          </>
        )}
      </div>
    </div>
  );
}
