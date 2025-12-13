import React from 'react';
import { LanguageCode, SUPPORTED_LANGUAGES } from '../../types/ai';
import './LanguageSelector.css';

interface LanguageSelectorProps {
  selectedLanguage: LanguageCode;
  onLanguageChange: (language: LanguageCode) => void;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  selectedLanguage,
  onLanguageChange,
}) => {
  return (
    <div className="language-selector">
      <label htmlFor="language-select" className="language-label">
        🌐 Language:
      </label>
      <select
        id="language-select"
        className="language-select"
        value={selectedLanguage}
        onChange={(e) => onLanguageChange(e.target.value as LanguageCode)}
      >
        {Object.entries(SUPPORTED_LANGUAGES).map(([code, name]) => (
          <option key={code} value={code}>
            {name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LanguageSelector;
