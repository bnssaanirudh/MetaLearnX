import React from 'react';
import * as Icons from './Icons';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          borderRadius: 'var(--radius-lg)',
          backgroundColor: 'var(--card-bg)',
          border: '1px solid rgba(255, 50, 50, 0.2)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1rem',
          height: '100%',
          minHeight: '200px'
        }}>
          <Icons.Zap size={32} color="#ff4444" />
          <h3 style={{ margin: 0, color: 'var(--text-primary)' }}>Component Rendering Halted</h3>
          <p style={{ margin: 0, color: 'var(--text-muted)', textAlign: 'center', maxWidth: '400px' }}>
            The visualization matrix encountered an unexpected topology error. Refresing the dashboard may resolve this state.
          </p>
          <button 
            className="btn btn-primary"
            onClick={() => window.location.reload()}
            style={{ marginTop: '1rem' }}
          >
            Reinitialize Context
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
