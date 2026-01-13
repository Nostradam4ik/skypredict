import React from 'react';

const PlaneLoader: React.FC = () => {
  return (
    <div className="plane-loader">
      <div className="loader-content">
        <div className="plane-animation">
          <svg
            className="plane-svg"
            viewBox="0 0 64 64"
            width="64"
            height="64"
          >
            <path
              fill="currentColor"
              d="M56.5 24.5l-12-12c-.4-.4-1-.6-1.5-.6h-6c-.5 0-1 .2-1.4.6l-3.6 3.6-10-5c-.3-.1-.6-.2-.9-.2h-5c-.8 0-1.5.5-1.8 1.2-.3.7-.1 1.5.4 2.1l8.8 8.8-8.8 8.8c-.5.5-.7 1.3-.4 2.1.3.7 1 1.2 1.8 1.2h5c.3 0 .6-.1.9-.2l10-5 3.6 3.6c.4.4.9.6 1.4.6h6c.6 0 1.1-.2 1.5-.6l12-12c.8-.8.8-2.1 0-2.8z"
            />
          </svg>
          <div className="cloud cloud-1"></div>
          <div className="cloud cloud-2"></div>
          <div className="cloud cloud-3"></div>
        </div>
        <div className="loader-text">
          <span className="loading-dots">Analyzing flight data</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill"></div>
        </div>
      </div>
    </div>
  );
};

export default PlaneLoader;
