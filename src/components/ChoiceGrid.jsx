// src/components/ChoiceGrid.jsx
// 2-column choice buttons.

export default function ChoiceGrid({ choices, onChoose, disabled }) {
  if (!choices || choices.length === 0) return null;

  return (
    <div className="choice-grid">
      {choices.map((label, i) => (
        <button
          key={i}
          className="choice-btn"
          disabled={disabled}
          onClick={() => onChoose(label)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
