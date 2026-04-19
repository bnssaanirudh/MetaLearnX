import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, LineChart, Line, Legend,
} from 'recharts';
import { api } from '../api/client';
import { Icons } from '../components/Icons';

function StatusBadge({ status }) {
  const styles = {
    running: { bg: 'var(--accent-light)', color: 'var(--accent)' },
    done: { bg: 'var(--bg-base)', color: 'var(--success)' },
    failed: { bg: '#fee2e2', color: 'var(--danger)' },
    pending: { bg: 'var(--bg-base)', color: 'var(--text-muted)' }
  };
  const current = styles[status] || styles.pending;

  return (
    <span className="badge" style={{ backgroundColor: current.bg, color: current.color, gap: 6 }}>
      {status === 'running' && <div className="spinner" style={{ width: 10, height: 10, borderWidth: 2 }} />}
      {status === 'done' && <Icons.CheckCircle size={12} />}
      {String(status).toUpperCase()}
    </span>
  );
}

export default function Optimization() {
  const [searchParams] = useSearchParams();
  const expId = searchParams.get('exp');
  const [selectedExp, setSelectedExp] = useState(expId || '');
  const qc = useQueryClient();

  const { data: history } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api.listExperiments(),
    refetchInterval: 5000,
  });
  const experiments = history?.experiments || [];

  const activeId = selectedExp || experiments[0]?.id;

  const { data: expData, isLoading: statusLoading } = useQuery({
    queryKey: ['exp-results', activeId],
    queryFn: () => api.getExpResults(activeId),
    enabled: !!activeId,
    refetchInterval: (data) => (data?.status === 'running' || data?.status === 'pending') ? 2000 : false,
  });

  const results = expData;
  const status = expData?.status || 'pending';
  const allTrials = results?.metrics?.all_trials || [];
  const paretoFront = results?.metrics?.pareto_front || [];
  const bestConfig = results?.pipeline_config || {};
  const bestMetrics = results?.metrics || {};
  const reasoningLogs = results?.reasoning_logs || [];

  const trialChartData = allTrials
    .filter(t => t.primary_metric > 0)
    .map((t, i) => ({
      trial: t.trial_number ?? i,
      score: t.primary_metric,
      model: t.model_name,
      time: t.train_time,
    }));

  const paretoData = paretoFront.map((p, i) => ({
    x: p.train_time || 0,
    y: p.primary_metric || 0,
    z: (p.model_size || 1000) / 1000,
    model: p.config?.model_name || '?',
    name: `Pareto ${i + 1}`,
  }));

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Bayesian Core Optimization</h1>
        <p className="page-subtitle">
          Hyperparameter synthesis utilizing Pareto-optimal search spaces and agentic multi-stage reasoning.
        </p>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        {/* Experiment selector */}
        <div className="card" style={{ marginBottom: 32 }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Icons.Database size={14} />
                Selected Research Run
              </label>
              <select
                className="form-select"
                style={{ padding: '12px 16px', fontWeight: 500 }}
                value={activeId || ''}
                onChange={(e) => setSelectedExp(e.target.value)}
              >
                {!activeId && <option value="">Select a run...</option>}
                {experiments.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.id.slice(0, 8)} &bull; {e.pipeline_config?.model_name || 'System Initializing...'}
                  </option>
                ))}
              </select>
            </div>
            {expData && <StatusBadge status={status} />}
          </div>
        </div>

        {/* Agentic Reasoning Manifold */}
        {(reasoningLogs.length > 0 || status === 'running') && (
          <div className="card" style={{ marginBottom: 32, borderLeft: '4px solid var(--primary)', background: '#f8fafc' }}>
            <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Icons.Shield size={18} />
              Autonomous Reasoning Manifold
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 16 }}>
              {reasoningLogs.map((log, idx) => (
                <div key={idx} style={{ 
                  padding: 16, 
                  background: 'white', 
                  borderRadius: 'var(--radius)', 
                  border: '1px solid var(--border)',
                  boxShadow: 'var(--shadow-sm)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, alignItems: 'center' }}>
                    <span style={{ 
                      fontSize: 10, 
                      fontWeight: 800, 
                      textTransform: 'uppercase', 
                      letterSpacing: '0.1em',
                      color: log.agent === 'Architect' ? 'var(--accent)' : 'var(--danger)',
                      background: log.agent === 'Architect' ? 'var(--accent-light)' : '#fee2e2',
                      padding: '4px 8px',
                      borderRadius: 4
                    }}>
                      {log.agent} &bull; Round {log.iteration + 1}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Thought Propagation Complete</span>
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6, fontWeight: 500 }}>
                    {log.reasoning}
                  </div>
                  {log.proposed_config && (
                    <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-secondary)', background: 'var(--bg-base)', padding: '8px 12px', borderRadius: 4, fontFamily: 'monospace' }}>
                      Pivoted Architecture: {log.proposed_config.model_name}
                    </div>
                  )}
                  {log.issue && (
                    <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                       <span className="badge" style={{ backgroundColor: '#fff7ed', color: '#c2410c', fontSize: '10px' }}>Issue: {log.issue}</span>
                       <span className="badge" style={{ backgroundColor: '#f0fdf4', color: '#15803d', fontSize: '10px' }}>Pivot: {log.refinement}</span>
                    </div>
                  )}
                </div>
              ))}
              {status === 'running' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px', color: 'var(--text-secondary)', fontSize: 13, fontStyle: 'italic' }}>
                  <div className="spinner" style={{ width: 14, height: 14 }} />
                  Processing next rationalization round...
                </div>
              )}
            </div>
          </div>
        )}

        {status === 'done' && results && (
          <>
            <div className="metrics-grid" style={{ marginBottom: 32 }}>
              <div className="metric-card">
                <div className="metric-label">Performance Signal</div>
                <div className="metric-value">{(bestMetrics.primary_metric || 0).toFixed(4)}</div>
                <div className="metric-sub">{bestMetrics.metric_name || 'Accuracy'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Topological Winner</div>
                <div className="metric-value" style={{ fontSize: 18 }}>{bestConfig.model_name || '—'}</div>
                <div className="metric-sub">Pipeline Core</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Operational Delta</div>
                <div className="metric-value">{(results.runtime_seconds || 0).toFixed(2)}s</div>
                <div className="metric-sub">Wall-clock Time</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Search Exhaustion</div>
                <div className="metric-value">{results.n_trials || 0}</div>
                <div className="metric-sub">Optuna Cycles</div>
              </div>
            </div>

            <div className="grid-2" style={{ marginBottom: 32 }}>
              <div className="card">
                <h3 className="card-title">
                  <Icons.Activity size={18} />
                  Convergence Trajectory
                </h3>
                <div style={{ height: 260, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trialChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="trial" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis domain={['auto', 'auto']} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip
                        contentStyle={{ borderRadius: '12px', border: '1px solid var(--border)', boxShadow: 'var(--shadow-lg)' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="var(--accent)"
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 6, fill: 'var(--accent)', stroke: 'white', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card">
                <h3 className="card-title">
                  <Icons.Target size={18} />
                  Pareto Dominance Surface
                </h3>
                <div style={{ height: 260, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis type="number" dataKey="x" name="Time" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis type="number" dataKey="y" name="Score" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <ZAxis type="number" dataKey="z" range={[60, 400]} />
                      <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const d = payload[0]?.payload;
                          return (
                            <div style={{ background: 'white', border: '1px solid var(--border)', borderRadius: '12px', padding: '12px 16px', boxShadow: 'var(--shadow-lg)', fontSize: 12 }}>
                              <div style={{ fontWeight: 800, color: 'var(--primary)', marginBottom: 4 }}>{d?.model}</div>
                              <div>Score: <strong>{d?.y?.toFixed(4)}</strong></div>
                              <div>Latency: <strong>{d?.x?.toFixed(3)}s</strong></div>
                            </div>
                          );
                        }}
                      />
                      <Scatter data={paretoData} fill="var(--accent)" fillOpacity={0.6} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="card-title">
                <Icons.Settings size={18} />
                Proposed Research Architecture
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 24, marginBottom: 32 }}>
                {Object.entries(bestConfig).filter(([k]) => k !== 'model_params').map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--primary)', marginTop: 2 }}>{String(v)}</div>
                  </div>
                ))}
              </div>

              <div style={{ padding: 24, background: '#f8fafc', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
                  {Object.entries(bestConfig.model_params || {}).map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{k}</div>
                      <div style={{ fontSize: 13, fontFamily: 'monospace', fontWeight: 700, color: 'var(--accent)' }}>{String(v)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {status === 'failed' && (
          <div className="alert alert-danger">
            <h4 style={{ margin: 0 }}>System Fault Detected</h4>
            <p style={{ margin: '8px 0 0', fontSize: 13 }}>{results?.error || 'Unknown architectural constraint violation.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
