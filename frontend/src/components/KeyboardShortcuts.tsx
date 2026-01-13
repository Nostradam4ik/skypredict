import React from 'react';
import { X, Keyboard } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface KeyboardShortcutsProps {
  onClose: () => void;
}

const KeyboardShortcuts: React.FC<KeyboardShortcutsProps> = ({ onClose }) => {
  const { t } = useTranslation();

  const shortcuts = [
    { keys: ['Enter'], description: t('pressToPredict') },
    { keys: ['T'], description: t('pressToToggleTheme') },
    { keys: ['C'], description: t('pressToCompare') },
    { keys: ['F'], description: t('pressToFullscreen') },
    { keys: ['Esc'], description: t('pressToClose') },
    { keys: ['?'], description: t('keyboardShortcuts') }
  ];

  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-modal" onClick={e => e.stopPropagation()}>
        <button className="shortcuts-close" onClick={onClose}>
          <X size={20} />
        </button>

        <h3>
          <Keyboard size={20} />
          {t('keyboardShortcuts')}
        </h3>

        <div className="shortcuts-list">
          {shortcuts.map((shortcut, index) => (
            <div key={index} className="shortcut-item">
              <div className="shortcut-keys">
                {shortcut.keys.map((key, i) => (
                  <kbd key={i}>{key}</kbd>
                ))}
              </div>
              <span className="shortcut-desc">{shortcut.description}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default KeyboardShortcuts;
