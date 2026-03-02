// src/components/ExploreMap.jsx
// SVG node map of the Withered Vale.

// Node positions (hand-tuned for the three MVP locations)
const NODE_POSITIONS = {
  'vale-crossroads': { x: 200, y: 150, label: 'The Crossroads' },
  'thornwatch-inn':  { x: 100, y: 320, label: 'Thornwatch Inn' },
  'old-mill':        { x: 320, y: 350, label: 'The Old Mill' },
};

// Connections (edges)
const EDGES = [
  ['vale-crossroads', 'thornwatch-inn'],
  ['vale-crossroads', 'old-mill'],
  ['thornwatch-inn', 'old-mill'],
];

export default function ExploreMap({
  worldData,
  currentLocationId,
  discoveredLocations,
  onSelectLocation,
}) {
  if (!worldData) return null;

  const isDiscovered = (id) => discoveredLocations.includes(id);

  return (
    <div className="explore-map-container">
      <h2 className="map-title">{worldData.region.name}</h2>
      <p className="map-subtitle">{worldData.region.description}</p>

      <svg viewBox="0 0 420 500" className="explore-map-svg">
        {/* Edges */}
        {EDGES.map(([a, b]) => {
          if (!isDiscovered(a) || !isDiscovered(b)) return null;
          const pa = NODE_POSITIONS[a];
          const pb = NODE_POSITIONS[b];
          return (
            <line
              key={`${a}-${b}`}
              x1={pa.x} y1={pa.y}
              x2={pb.x} y2={pb.y}
              className="map-edge"
            />
          );
        })}

        {/* Nodes */}
        {Object.entries(NODE_POSITIONS).map(([id, pos]) => {
          if (!isDiscovered(id)) return null;
          const isCurrent = id === currentLocationId;
          return (
            <g
              key={id}
              className={`map-node${isCurrent ? ' current' : ''}`}
              onClick={() => onSelectLocation(id)}
              style={{ cursor: 'pointer' }}
            >
              <circle cx={pos.x} cy={pos.y} r={isCurrent ? 18 : 14} />
              {isCurrent && (
                <circle cx={pos.x} cy={pos.y} r={24} className="map-node-pulse" />
              )}
              <text x={pos.x} y={pos.y + 36} textAnchor="middle">
                {pos.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
