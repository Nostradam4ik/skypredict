import React from 'react';
import { useTranslation } from 'react-i18next';
import { Globe } from 'lucide-react';

const languages = [
  { code: 'en', name: 'English', flag: '🇬🇧' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'es', name: 'Español', flag: '🇪🇸' }
];

const LanguageSelector: React.FC = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = React.useState(false);

  const currentLang = languages.find(l => l.code === i18n.language) || languages[0];

  const changeLanguage = (code: string) => {
    i18n.changeLanguage(code);
    localStorage.setItem('skypredict-language', code);
    setIsOpen(false);
  };

  return (
    <div className="language-selector">
      <button
        className="language-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Change language"
      >
        <Globe size={18} />
        <span className="lang-flag">{currentLang.flag}</span>
      </button>

      {isOpen && (
        <>
          <div className="language-backdrop" onClick={() => setIsOpen(false)} />
          <div className="language-dropdown">
            {languages.map((lang) => (
              <button
                key={lang.code}
                className={`language-option ${lang.code === i18n.language ? 'active' : ''}`}
                onClick={() => changeLanguage(lang.code)}
              >
                <span className="lang-flag">{lang.flag}</span>
                <span className="lang-name">{lang.name}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default LanguageSelector;
