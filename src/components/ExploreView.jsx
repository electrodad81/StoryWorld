// src/components/ExploreView.jsx
// Exploration mode root — switches between map, location detail, and NPC conversation.

import ExploreMap from './ExploreMap.jsx';
import LocationView from './LocationView.jsx';
import NPCConversation from './NPCConversation.jsx';

export default function ExploreView({ explore, onMenuToggle }) {
  const { explorePhase } = explore;

  return (
    <div className="explore-viewport">
      {/* Top bar — same pattern as story mode */}
      <div className="top-bar">
        <button className="menu-btn" onClick={onMenuToggle}>☰</button>
        <span className="beat-indicator">
          {explorePhase === 'map' ? 'World Map' :
           explorePhase === 'conversation' ? explore.activeNpc?.name :
           explore.getCurrentLocation()?.name || 'Exploring'}
        </span>
      </div>

      {explorePhase === 'map' && (
        <ExploreMap
          worldData={explore.worldData}
          currentLocationId={explore.currentLocationId}
          discoveredLocations={explore.discoveredLocations}
          onSelectLocation={explore.travelTo}
        />
      )}

      {explorePhase === 'location' && (
        <LocationView
          location={explore.getCurrentLocation()}
          npcs={explore.getNpcsAtLocation()}
          connections={explore.getConnectedLocations()}
          npcDispositions={explore.npcDispositions}
          onEngageNpc={explore.engageNpc}
          onTravel={explore.travelTo}
          onOpenMap={explore.openMap}
        />
      )}

      {explorePhase === 'conversation' && (
        <NPCConversation
          npc={explore.activeNpc}
          disposition={explore.npcDispositions[explore.activeNpc?.id] || 50}
          conversationHistory={explore.conversationHistory}
          isStreaming={explore.isStreaming}
          streamedText={explore.streamedText}
          interactionOptions={explore.interactionOptions}
          isGenerating={explore.isGenerating}
          onTalk={explore.talkToNpc}
          onLeave={explore.leaveConversation}
        />
      )}
    </div>
  );
}
