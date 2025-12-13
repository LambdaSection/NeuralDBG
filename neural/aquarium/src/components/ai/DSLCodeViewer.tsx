import React from 'react';
import './DSLCodeViewer.css';

interface DSLCodeViewerProps {
  code: string;
  isEditing: boolean;
  onEdit: () => void;
  onSave: () => void;
  onCancel: () => void;
  onApply: () => void;
  onChange: (code: string) => void;
}

const DSLCodeViewer: React.FC<DSLCodeViewerProps> = ({
  code,
  isEditing,
  onEdit,
  onSave,
  onCancel,
  onApply,
  onChange,
}) => {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      alert('DSL code copied to clipboard!');
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model.neural';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="dsl-code-viewer">
      <div className="viewer-header">
        <h3>Generated DSL Code</h3>
        <div className="viewer-actions">
          {!isEditing ? (
            <>
              <button
                className="action-button edit-button"
                onClick={onEdit}
                title="Edit code"
              >
                ✏️ Edit
              </button>
              <button
                className="action-button copy-button"
                onClick={handleCopy}
                title="Copy to clipboard"
              >
                📋 Copy
              </button>
              <button
                className="action-button download-button"
                onClick={handleDownload}
                title="Download as file"
              >
                ⬇️ Download
              </button>
              <button
                className="action-button apply-button"
                onClick={onApply}
                title="Apply to model"
              >
                ✓ Apply
              </button>
            </>
          ) : (
            <>
              <button
                className="action-button save-button"
                onClick={onSave}
                title="Save changes"
              >
                💾 Save
              </button>
              <button
                className="action-button cancel-button"
                onClick={onCancel}
                title="Cancel editing"
              >
                ✕ Cancel
              </button>
            </>
          )}
        </div>
      </div>

      <div className="code-container">
        {isEditing ? (
          <textarea
            className="code-editor"
            value={code}
            onChange={(e) => onChange(e.target.value)}
            spellCheck={false}
          />
        ) : (
          <pre className="code-display">
            <code>{code}</code>
          </pre>
        )}
      </div>

      <div className="viewer-footer">
        <span className="code-stats">
          Lines: {code.split('\n').length} | Characters: {code.length}
        </span>
      </div>
    </div>
  );
};

export default DSLCodeViewer;
