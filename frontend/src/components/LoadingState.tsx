import React from 'react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading simulation data...',
}) => {
  return (
    <div className="loading-container">
      <div className="cricket-ball-spinner">
        <div className="seam"></div>
      </div>
      <p className="loading-text">{message}</p>
    </div>
  );
};
