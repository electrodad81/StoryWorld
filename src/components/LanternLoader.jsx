// src/components/LanternLoader.jsx
// CSS-animated lantern loader with flame.

export default function LanternLoader({ caption = 'Working\u2026' }) {
  return (
    <div className="lantern-loader">
      <div className="lantern">
        <div className="handle" />
        <div className="body">
          <span className="flame" />
        </div>
      </div>
      {caption && <div className="lantern-caption">{caption}</div>}
    </div>
  );
}
