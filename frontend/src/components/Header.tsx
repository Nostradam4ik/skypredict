import React from 'react';
import { useTranslation } from 'react-i18next';
import { Plane, Activity, GitCompare } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

interface HeaderProps {
  apiStatus: 'connected' | 'disconnected' | 'checking';
  isDarkMode: boolean;
  onThemeToggle: () => void;
  compareMode: boolean;
  onCompareModeToggle: () => void;
  compareCount: number;
}

const Header: React.FC<HeaderProps> = ({
  apiStatus,
  isDarkMode,
  onThemeToggle,
  compareMode,
  onCompareModeToggle,
  compareCount
}) => {
  const { t } = useTranslation();

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <Plane size={32} />
          <h1>{t('title')}</h1>
        </div>
        <p className="tagline">{t('tagline')}</p>
      </div>

      <div className="header-actions">
        <button
          className={`compare-mode-btn ${compareMode ? 'active' : ''}`}
          onClick={onCompareModeToggle}
          title={t('compare')}
        >
          <GitCompare size={18} />
          <span>{t('compare')}</span>
          {compareCount > 0 && (
            <span className="compare-badge">{compareCount}</span>
          )}
        </button>

        <ThemeToggle isDark={isDarkMode} onToggle={onThemeToggle} />

        <div className={`api-status ${apiStatus}`}>
          <Activity size={16} />
          <span>
            {apiStatus === 'connected' && t('apiConnected')}
            {apiStatus === 'disconnected' && t('apiDisconnected')}
            {apiStatus === 'checking' && t('checking')}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
