// src/components/NPCConversation.jsx
// NPC dialogue view with streaming responses, topic/quest options, and exchange limits.

import StreamingText from './StreamingText.jsx';
import LanternLoader from './LanternLoader.jsx';

/** Strip curly-brace JSON and other LLM artifacts from an option label. */
function sanitizeLabel(raw) {
  if (!raw || typeof raw !== 'string') return 'Make small talk';
  let s = raw.trim();
  // If it looks like a JSON object, try to extract a string value
  if (s.startsWith('{')) {
    try {
      const obj = JSON.parse(s);
      const val = Object.values(obj).find(v => typeof v === 'string');
      if (val) return val.replace(/\.$/, '');
    } catch { /* fall through */ }
    // Brute-force: strip everything up to first quote pair
    const m = s.match(/"([^"]+)"/);
    if (m) return m[1].replace(/\.$/, '');
    return 'Make small talk';
  }
  // Strip markdown fences
  s = s.replace(/^```[\s\S]*?```$/g, '').trim();
  // Strip wrapping quotes
  s = s.replace(/^["']|["']$/g, '').trim();
  return s || 'Make small talk';
}

export default function NPCConversation({
  npc,
  location,
  disposition,
  conversationHistory,
  isStreaming,
  streamedText,
  interactionOptions,
  isGenerating,
  onTalk,
  onLeave,
  exchangeCount = 0,
  maxExchanges = 4,
  questNotification,
  nothingNewMessage,
}) {
  if (!npc) return null;

  const dispositionLabel =
    disposition >= 70 ? 'Friendly' :
    disposition >= 50 ? 'Neutral' :
    disposition >= 30 ? 'Wary' : 'Hostile';

  // Build scene-setting context line from location + NPC
  const contextLine = location
    ? `${location.description} ${npc.name} is here — ${npc.archetype}.`
    : `${npc.name} — ${npc.archetype}.`;

  // Show the latest NPC response (either streaming or from history)
  const latestNpcMessage = isStreaming
    ? streamedText
    : conversationHistory.filter(m => m.role === 'assistant').pop()?.content || npc.greeting;

  // Dismissed — conversation over after max exchanges
  const isDismissed = exchangeCount >= maxExchanges && !isStreaming;

  // Whether we're still on the opening (no player messages yet)
  const isOpening = conversationHistory.filter(m => m.role === 'user').length === 0;

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

      {/* Quest notification */}
      {questNotification && (
        <div className="quest-notification">
          {questNotification.type === 'completed'
            ? `Quest Complete: ${questNotification.quest.title}`
            : `Quest Accepted: ${questNotification.quest.title}`
          }
        </div>
      )}

      {/* Nothing new state */}
      {nothingNewMessage ? (
        <div className="npc-dialogue">
          <div className="npc-context-line">{contextLine}</div>
          <div className="npc-nothing-new">&ldquo;{nothingNewMessage}&rdquo;</div>
        </div>
      ) : (
        <>
          {/* Dialogue area */}
          <div className="npc-dialogue">
            {/* Scene context — show on opening and first exchange */}
            {isOpening && (
              <div className="npc-context-line">{contextLine}</div>
            )}
            {latestNpcMessage ? (
              <div className="npc-speech">
                <StreamingText
                  text={latestNpcMessage}
                  isStreaming={isStreaming}
                  quoted={true}
                />
              </div>
            ) : isGenerating ? (
              <LanternLoader caption={`${npc.name} considers you…`} />
            ) : null}
          </div>

          {/* Exchange counter */}
          {exchangeCount > 0 && !isDismissed && (
            <div className="exchange-counter">
              {exchangeCount} / {maxExchanges} exchanges
            </div>
          )}

          {/* Interaction options */}
          <div className="npc-options">
            {isDismissed ? (
              <button
                className="choice-btn leave-btn"
                onClick={onLeave}
              >
                Walk away
              </button>
            ) : isGenerating && !isStreaming ? (
              <LanternLoader caption="Considering your options…" />
            ) : (
              <>
                {interactionOptions.map((option, i) => {
                  const optionType = typeof option === 'string' ? 'flavor' : option.type;
                  const rawLabel = typeof option === 'string' ? option : option.label;
                  const label = sanitizeLabel(rawLabel);
                  const btnClass = optionType === 'topic' ? 'choice-btn topic-option' :
                                   optionType === 'quest' || optionType === 'cross_quest' ? 'choice-btn quest-option' :
                                   'choice-btn';
                  return (
                    <button
                      key={i}
                      className={btnClass}
                      onClick={() => onTalk(option)}
                      disabled={isStreaming || isGenerating}
                    >
                      {label}
                    </button>
                  );
                })}
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
        </>
      )}

      {/* Walk away from nothing-new state */}
      {nothingNewMessage && (
        <div className="npc-options">
          <button className="choice-btn leave-btn" onClick={onLeave}>
            Walk away
          </button>
        </div>
      )}
    </div>
  );
}
