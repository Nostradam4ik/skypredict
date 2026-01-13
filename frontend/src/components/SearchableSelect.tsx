import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown } from 'lucide-react';

interface SearchableSelectProps {
  options: Record<string, string>;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  icon?: React.ReactNode;
}

const SearchableSelect: React.FC<SearchableSelectProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Search...',
  label,
  icon
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredOptions = Object.entries(options).filter(([code, name]) =>
    code.toLowerCase().includes(search.toLowerCase()) ||
    name.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearch('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSelect = (code: string) => {
    onChange(code);
    setIsOpen(false);
    setSearch('');
  };

  return (
    <div className="searchable-select" ref={containerRef}>
      {label && (
        <label className="select-label">
          {icon} {label}
        </label>
      )}
      <div
        className={`select-trigger ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="selected-value">
          {value ? `${value} - ${options[value]}` : placeholder}
        </span>
        <ChevronDown size={16} className={`chevron ${isOpen ? 'rotated' : ''}`} />
      </div>

      {isOpen && (
        <div className="select-dropdown">
          <div className="search-input-wrapper">
            <Search size={16} />
            <input
              ref={inputRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={placeholder}
              className="search-input"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
          <div className="options-list">
            {filteredOptions.length > 0 ? (
              filteredOptions.map(([code, name]) => (
                <div
                  key={code}
                  className={`option ${code === value ? 'selected' : ''}`}
                  onClick={() => handleSelect(code)}
                >
                  <span className="option-code">{code}</span>
                  <span className="option-name">{name}</span>
                </div>
              ))
            ) : (
              <div className="no-results">No results found</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchableSelect;
