// src/components/LocationView.jsx
// Location detail view: illustration, description, NPC list, travel options.

export default function LocationView({
  location,
  npcs,
  connections,
  npcDispositions,
  onEngageNpc,
  onTravel,
  onOpenMap,
}) {
  if (!location) return null;

  return (
    <div className="location-view">
      {/* Location illustration header */}
      <div className="location-header">
        {location.coverImage && (
          <img src={location.coverImage} alt={location.name} className="location-cover" />
        )}
        <div className="location-header-overlay">
          <h2>{location.name}</h2>
        </div>
      </div>

      {/* Description */}
      <div className="location-body">
        <p className="location-description">{location.description}</p>

        {/* NPCs */}
        {npcs.length > 0 && (
          <div className="location-section">
            <h3>People Here</h3>
            {npcs.map(npc => {
              const disposition = npcDispositions[npc.id] || npc.baseDisposition;
              const dispositionLabel =
                disposition >= 70 ? 'Friendly' :
                disposition >= 50 ? 'Neutral' :
                disposition >= 30 ? 'Wary' : 'Hostile';
              return (
                <button
                  key={npc.id}
                  className="npc-card"
                  onClick={() => onEngageNpc(npc.id)}
                >
                  {npc.portrait && (
                    <img src={npc.portrait} alt={npc.name} className="npc-portrait" />
                  )}
                  <div className="npc-card-info">
                    <span className="npc-name">{npc.name}</span>
                    <span className="npc-archetype">{npc.archetype}</span>
                  </div>
                  <span className={`npc-disposition ${dispositionLabel.toLowerCase()}`}>
                    {dispositionLabel}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {/* Travel options */}
        <div className="location-section">
          <h3>Paths</h3>
          {connections.map(loc => (
            <button
              key={loc.id}
              className="travel-btn"
              onClick={() => onTravel(loc.id)}
            >
              Travel to {loc.name}
            </button>
          ))}
        </div>

        {/* Back to map */}
        <button className="map-btn" onClick={onOpenMap}>
          Open Map
        </button>
      </div>
    </div>
  );
}
