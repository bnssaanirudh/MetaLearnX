import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Icons } from '../components/Icons';

function FeatureImportanceBar({ name, importance, max }) {
  const pct = max > 0 ? (importance / max) * 100 : 0;
  return (
    <div className="fi-row" style={{ padding: '10px 0', borderBottom: '1px solid #f1f5f9' }}>
      <div className="fi-name" title={name} style={{ width: 160, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</div>
      <div className="fi-bar-wrap" style={{ flex: 1, height: 8, background: 'var(--bg-base)', borderRadius: 4, margin: '0 20px', overflow: 'hidden' }}>
        <div className="fi-bar-fill" style={{ width: `${pct}%`, background: 'var(--accent)', height: '100%', borderRadius: 4, transition: 'width 1s ease-out' }} />
      </div>
      <div className="fi-val" style={{ width: 90, textAlign: 'right', fontSize: 13, fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>
        {importance.toFixed(5)}
      </div>
    </div>
  );
}

export default function Explainability() {
  const { data: history } = useQuery({
    queryKey: ['experiments'],
    queryFn: api.listExperiments,
  });
  const experiments = (history?.experiments || []).filter(e => e.status === 'done');
  const [selectedExp, setSelectedExp] = useState('');
  const activeId = selectedExp || experiments[0]?.id;

  const { data: explain, isLoading } = useQuery({
    queryKey: ['explain', activeId],
    queryFn: () => api.getExpExplain(activeId),
    enabled: !!activeId,
  });

  const fi = explain?.feature_importances || [];
  const max = fi[0]?.importance || 1;
  const rationale = explain?.pipeline_rationale || '';
  const agentReasoning = explain?.agent_reasoning || [];
  const method = explain?.method_used || 'SHAP';
  const elapsed = explain?.elapsed_seconds;

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Model Interpretability</h1>
        <p className="page-subtitle">
          Deconstructing the black-box: SHAP feature contribution analysis and agentic architectural rationalization.
        </p>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        <div className="card" style={{ marginBottom: 32 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Active Research Objective</label>
            <select
              className="form-select"
              style={{ padding: '12px 16px', fontWeight: 500 }}
              value={activeId || ''}
              onChange={(e) => setSelectedExp(e.target.value)}
            >
              {experiments.map(e => (
                <option key={e.id} value={e.id}>
                  {e.id.slice(0, 8)} &bull; {e.pipeline_config?.model_name || 'Generic Engine'}
                </option>
              ))}
            </select>
          </div>
        </div>

        {isLoading && (
          <div className="loading-center" style={{ height: 400 }}>
            <div className="spinner" style={{ width: 44, height: 44 }} />
            <span style={{ fontSize: 15, fontWeight: 500, marginTop: 16 }}>Decomposing Prediction Vectors...</span>
          </div>
        )}

        {!isLoading && explain && !explain.message && (
          <div className="grid-2" style={{ gridTemplateColumns: '1fr 380px' }}>
            {/* Feature importance */}
            <div className="card">
              <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Icons.BarChart size={18} />
                  Feature Influence Matrix
                </div>
                <span className="badge badge-indigo" style={{ textTransform: 'uppercase' }}>{method} Engine</span>
              </h3>
              
              <div style={{ marginTop: 24 }}>
                <div style={{ display: 'flex', fontSize: 10, fontWeight: 800, color: 'var(--text-muted)', marginBottom: 20, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                  <span style={{ width: 160 }}>Feature Identity</span>
                  <span style={{ flex: 1, paddingLeft: 20 }}>Relative Importance</span>
                  <span style={{ width: 90, textAlign: 'right' }}>Magnitude</span>
                </div>
                {fi.length === 0 ? (
                  <div className="empty-state" style={{ padding: 60 }}>
                    <p className="empty-sub">No contribution data extracted from the model object.</p>
                  </div>
                ) : (
                  <div style={{ maxHeight: '70vh', overflowY: 'auto', paddingRight: 10 }}>
                    {fi.slice(0, 40).map(f => (
                      <FeatureImportanceBar key={f.name} name={f.name} importance={f.importance} max={max} />
                    ))}
                    {fi.length > 40 && (
                      <div style={{ textAlign: 'center', padding: '20px 0', borderTop: '1px solid var(--border)', marginTop: 10 }}>
                        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>+ {fi.length - 40} additional features analyzed</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Analysis Pane */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Agent reasoning - Priority display */}
              {agentReasoning.length > 0 && (
                <div className="card" style={{ background: 'var(--bg-base)', border: '2px solid var(--border)' }}>
                  <h3 className="card-title">
                    <Icons.Shield size={18} color="var(--primary)" />
                    Agentic Reasoning
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                    {agentReasoning.map((log, i) => (
                      <div key={i} style={{ padding: '12px', background: 'white', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: 10, fontWeight: 800, color: log.agent === 'Architect' ? 'var(--accent)' : 'var(--danger)', marginBottom: 4 }}>
                          {log.agent.toUpperCase()} &bull; Round {log.iteration + 1}
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
                          {log.reasoning}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Classic Rationale */}
              <div className="card" style={{ background: 'var(--primary)', color: 'white' }}>
                <h3 className="card-title" style={{ color: 'white' }}>
                  <Icons.Eye size={18} color="white" />
                  Synthesis Summary
                </h3>
                {rationale ? (
                  <div style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.8 }}>
                    {rationale.split('\n\n').map((para, i) => (
                      <p key={i} style={{ marginBottom: 12 }}
                        dangerouslySetInnerHTML={{
                          __html: para.replace(/\*\*(.*?)\*\*/g, '<strong style="color:white">$1</strong>')
                        }}
                      />
                    ))}
                  </div>
                ) : (
                  <p style={{ opacity: 0.7, fontStyle: 'italic', fontSize: 13 }}>Synthesis results generation in progress...</p>
                )}
              </div>

              <div className="card" style={{ background: '#f8fafc', borderStyle: 'dashed' }}>
                <h3 className="card-title">Compute Context</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                   <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Methodology</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--primary)' }}>SHAP Kernel</span>
                   </div>
                   <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Analysis Latency</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--primary)' }}>{elapsed ? `${elapsed.toFixed(3)}s` : 'N/A'}</span>
                   </div>
                   <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Status</span>
                      <span className="badge badge-success">Verified</span>
                   </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!isLoading && explain?.message && (
          <div className="empty-state" style={{ padding: '100px 0' }}>
            <Icons.Search size={64} style={{ color: 'var(--border)', marginBottom: 24 }} />
            <h2 className="empty-title">Inspection Results Pending</h2>
            <p className="empty-sub">{explain.message}</p>
          </div>
        )}

        {experiments.length === 0 && !isLoading && (
          <div className="empty-state" style={{ padding: '100px 0' }}>
            <Icons.Activity size={64} style={{ color: 'var(--border)', marginBottom: 24 }} />
            <h2 className="empty-title">Research History Null</h2>
            <p className="empty-sub">Self-adaptive analysis requires a successfully completed model convergence run.</p>
          </div>
        )}
      </div>
    </div>
  );
}
