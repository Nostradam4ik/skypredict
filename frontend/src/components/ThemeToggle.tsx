import React from 'react';
import { Sun, Moon } from 'lucide-react';

interface ThemeToggleProps {
  isDark: boolean;
  onToggle: () => void;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDark, onToggle }) => {
  return (
    <button
      className={`theme-toggle ${isDark ? 'dark' : 'light'}`}
      onClick={onToggle}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <div className="toggle-track">
        <Sun size={14} className="sun-icon" />
        <Moon size={14} className="moon-icon" />
        <div className="toggle-thumb" />
      </div>
    </button>
  );
};

export default ThemeToggle;
