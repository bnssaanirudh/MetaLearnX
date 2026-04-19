import React from 'react';
import { Icons } from './Icons';

const AgenticFlowDiagram = ({ activeState = 'all' }) => {
  // activeState: 'idle' | 'profiling' | 'architecting' | 'optimizing' | 'critic' | 'done' | 'all'
  
  // Helper to determine if a node should pulse
  const isPulsing = (targetState) => activeState === targetState || activeState === 'all';
  
  // Helper to determine if a path should be animated
  const isPathAnimated = (targetState) => activeState === targetState || activeState === 'all';
  
  // Helper to determine if a node has been passed (to give it a solid color instead of gray)
  const isPassed = (targetState) => {
    if (activeState === 'all') return true;
    const states = ['idle', 'profiling', 'architecting', 'optimizing', 'critic', 'done'];
    const currentIndex = states.indexOf(activeState);
    const targetIndex = states.indexOf(targetState);
    return currentIndex >= targetIndex;
  };

  const grayStroke = "#e2e8f0";
  
  return (
    <div style={{
      width: '100%',
      maxWidth: 900,
      margin: '0 auto',
      background: 'white',
      borderRadius: 'var(--radius-xl)',
      padding: '40px',
      boxShadow: '0 20px 40px -10px rgba(0,0,0,0.08)',
      border: '1px solid var(--border)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background Graphic elements for pop */}
      <div style={{ position: 'absolute', top: -50, right: -50, width: 200, height: 200, background: 'var(--gradient-primary)', opacity: isPulsing('architecting') ? 0.1 : 0.02, filter: 'blur(50px)', borderRadius: '50%', transition: 'all 1s' }} />
      <div style={{ position: 'absolute', bottom: -50, left: -50, width: 200, height: 200, background: 'var(--gradient-secondary)', opacity: isPulsing('critic') ? 0.1 : 0.02, filter: 'blur(50px)', borderRadius: '50%', transition: 'all 1s' }} />

      <svg width="100%" height="320" viewBox="0 0 800 320" style={{ overflow: 'visible' }}>
        <defs>
          <linearGradient id="gradientPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FF0080" />
            <stop offset="100%" stopColor="#7928CA" />
          </linearGradient>
          <linearGradient id="gradientSecondary" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00DFD8" />
            <stop offset="100%" stopColor="#007CF0" />
          </linearGradient>
          <marker id="arrowHeadPrimary" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill="url(#gradientPrimary)" />
          </marker>
          <marker id="arrowHeadSecondary" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill="url(#gradientSecondary)" />
          </marker>
          <marker id="arrowHeadGray" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill={grayStroke} />
          </marker>
        </defs>

        {/* --- Path Lines --- */}
        
        {/* Upload -> Profiler */}
        <path d="M 120 160 L 220 90" fill="none" strokeWidth="3" 
          stroke={isPassed('profiling') ? "url(#gradientPrimary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('profiling') ? "path-animated" : ""} 
          markerEnd={isPassed('profiling') ? "url(#arrowHeadPrimary)" : "url(#arrowHeadGray)"} 
        />
        
        {/* Profiler -> Architect */}
        <path d="M 280 90 L 280 120" fill="none" strokeWidth="3" 
          stroke={isPassed('architecting') ? "url(#gradientPrimary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('architecting') ? "path-animated" : ""} 
          markerEnd={isPassed('architecting') ? "url(#arrowHeadPrimary)" : "url(#arrowHeadGray)"} 
        />

        {/* Upload -> Architect (Bypass Profiler theoretically, just visual) */}
        <path d="M 120 160 L 240 160" fill="none" strokeWidth="3" 
          stroke={isPassed('architecting') ? "url(#gradientPrimary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('architecting') ? "path-animated" : ""} 
          markerEnd={isPassed('architecting') ? "url(#arrowHeadPrimary)" : "url(#arrowHeadGray)"} 
        />

        {/* Architect -> Optimizer */}
        <path d="M 320 160 L 460 160" fill="none" strokeWidth="3" 
          stroke={isPassed('optimizing') ? "url(#gradientPrimary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('optimizing') ? "path-animated" : ""} 
          markerEnd={isPassed('optimizing') ? "url(#arrowHeadPrimary)" : "url(#arrowHeadGray)"} 
        />

        {/* Optimizer -> Critic */}
        <path d="M 500 200 L 500 250 L 440 250" fill="none" strokeWidth="3" 
          stroke={isPassed('critic') ? "url(#gradientSecondary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('critic') ? "path-animated-secondary" : ""} 
          markerEnd={isPassed('critic') ? "url(#arrowHeadSecondary)" : "url(#arrowHeadGray)"} 
        />

        {/* Critic -> Optimizer (The Feedback Loop) */}
        <path d="M 360 250 L 320 250 L 320 200 L 460 180" fill="none" strokeWidth="3" 
          stroke={isPassed('critic') ? "url(#gradientSecondary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('critic') ? "path-animated-secondary" : ""} 
          markerEnd={isPassed('critic') ? "url(#arrowHeadSecondary)" : "url(#arrowHeadGray)"} 
        />

        {/* Optimizer -> Explain */}
        <path d="M 540 160 L 680 160" fill="none" strokeWidth="3" 
          stroke={isPassed('done') ? "url(#gradientPrimary)" : grayStroke} 
          strokeDasharray="12 12" 
          className={isPathAnimated('done') ? "path-animated" : ""} 
          markerEnd={isPassed('done') ? "url(#arrowHeadPrimary)" : "url(#arrowHeadGray)"} 
        />

        {/* --- Render Nodes --- */}
        {/* 1. Dataset Node */}
        <g transform="translate(60, 160)" style={{ transition: 'all 0.5s' }}>
          <circle cx="0" cy="0" r="40" fill="white" 
            stroke={isPassed('idle') ? "var(--primary)" : grayStroke} strokeWidth="2" 
            className={isPulsing('idle') ? "node-pulse" : ""} 
          />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('idle') ? 'var(--primary)' : 'var(--text-muted)' }}>
                <Icons.Database size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="700" fill={isPassed('idle') ? "var(--primary)" : "var(--text-muted)"}>Raw Dataset</text>
        </g>

        {/* 2. Profiler Node */}
        <g transform="translate(280, 50)" style={{ transition: 'all 0.5s' }}>
          <circle cx="0" cy="0" r="40" fill="white" 
            stroke={isPassed('profiling') ? "url(#gradientSecondary)" : grayStroke} strokeWidth="3" 
            className={isPulsing('profiling') ? "node-pulse" : ""} 
          />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('profiling') ? '#007CF0' : 'var(--text-muted)' }}>
                <Icons.Activity size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="700" fill={isPassed('profiling') ? "var(--text-secondary)" : "var(--text-muted)"}>Data Intelligence</text>
        </g>

        {/* 3. Architect Node */}
        <g transform="translate(280, 160)" style={{ transition: 'all 0.5s' }}>
          <circle cx="0" cy="0" r="40" fill="white" 
            stroke={isPassed('architecting') ? "url(#gradientPrimary)" : grayStroke} strokeWidth="3" 
            className={isPulsing('architecting') ? "node-pulse" : ""} 
          />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('architecting') ? '#FF0080' : 'var(--text-muted)' }}>
                <Icons.Cpu size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="800" fill={isPassed('architecting') ? "var(--primary)" : "var(--text-muted)"}>Architect Agent</text>
        </g>

        {/* 4. Optimizer Node */}
        <g transform="translate(500, 160)" style={{ transition: 'all 0.5s' }}>
          <rect x="-40" y="-40" width="80" height="80" rx="16" fill={isPassed('optimizing') ? "var(--primary)" : "white"} stroke={isPassed('optimizing') ? "none" : grayStroke} strokeWidth="2" />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('optimizing') ? 'white' : 'var(--text-muted)' }}>
                <Icons.Target size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="800" fill={isPassed('optimizing') ? "var(--primary)" : "var(--text-muted)"}>Optuna Core</text>
          <text x="0" y="76" textAnchor="middle" fontSize="10" fontWeight="600" fill="var(--text-muted)">NSGA-II Search</text>
          {isPulsing('optimizing') && activeState !== 'all' && (
             <circle cx="0" cy="0" r="45" fill="none" stroke="var(--primary)" strokeWidth="2" strokeDasharray="4 4" className="path-animated" />
          )}
        </g>

        {/* 5. Critic Node */}
        <g transform="translate(400, 250)" style={{ transition: 'all 0.5s' }}>
          <circle cx="0" cy="0" r="40" fill="white" 
            stroke={isPassed('critic') ? "url(#gradientSecondary)" : grayStroke} strokeWidth="3" 
            className={isPulsing('critic') ? "node-pulse" : ""} 
          />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('critic') ? '#007CF0' : 'var(--text-muted)' }}>
                <Icons.Shield size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="800" fill={isPassed('critic') ? "var(--primary)" : "var(--text-muted)"}>Critic Agent</text>
        </g>

        {/* 6. Output Node */}
        <g transform="translate(720, 160)" style={{ transition: 'all 0.5s' }}>
          <circle cx="0" cy="0" r="40" fill="white" 
            stroke={isPassed('done') ? "var(--primary)" : grayStroke} strokeWidth="2" 
            className={isPulsing('done') ? "node-pulse" : ""} 
          />
          <foreignObject x="-24" y="-24" width="48" height="48">
             <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: isPassed('done') ? 'var(--primary)' : 'var(--text-muted)' }}>
                <Icons.Eye size={24} />
             </div>
          </foreignObject>
          <text x="0" y="60" textAnchor="middle" fontSize="12" fontWeight="700" fill={isPassed('done') ? "var(--primary)" : "var(--text-muted)"}>Interpretability</text>
        </g>

      </svg>
    </div>
  );
};

export default React.memo(AgenticFlowDiagram);
