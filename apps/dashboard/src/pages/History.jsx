import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { Icons } from '../components/Icons';

function StatusBadge({ status }) {
  const styles = {
    running: { bg: 'var(--accent-light)', color: 'var(--accent)' },
    done: { bg: '#dcfce7', color: 'var(--success)' },
    failed: { bg: '#fee2e2', color: 'var(--danger)' },
    pending: { bg: 'var(--bg-base)', color: 'var(--text-muted)' }
  };
  const current = styles[status] || styles.pending;

  return (
    <span className="badge" style={{ backgroundColor: current.bg, color: current.color, fontSize: '10px', textTransform: 'uppercase' }}>
      {status}
    </span>
  );
}

export default function History() {
  const navigate = useNavigate();
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['experiments'],
    queryFn: api.listExperiments,
    refetchInterval: 5000,
  });
  const experiments = data?.experiments || [];

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 className="page-title">Run Repository</h1>
            <p className="page-subtitle">Centralized observation and status tracking for all meta-learning experiments.</p>
          </div>
          <button className="btn btn-outline" onClick={() => refetch()}>
            <Icons.Activity size={16} />
            Synch Records
          </button>
        </div>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        {isLoading ? (
          <div className="loading-center" style={{ height: 300 }}>
            <div className="spinner" />
            <span style={{ fontSize: 14 }}>Retrieving Experiment Meta-Data...</span>
          </div>
        ) : experiments.length === 0 ? (
          <div className="empty-state" style={{ padding: '80px 0' }}>
            <Icons.Database size={64} style={{ color: 'var(--border)', marginBottom: 24 }} />
            <h2 className="empty-title">Repository Empty</h2>
            <p className="empty-sub">Initiate a research experiment via the Ingestion module to populate history.</p>
          </div>
        ) : (
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div className="table-wrapper">
              <table style={{ border: 'none' }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '16px 24px' }}>Asset ID</th>
                    <th>Status</th>
                    <th>Topological Candidate</th>
                    <th>Performance</th>
                    <th>Latency</th>
                    <th>Trials</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((e) => {
                    const metrics = e.metrics || {};
                    const cfg = e.pipeline_config || {};
                    return (
                      <tr key={e.id}>
                        <td className="td-mono" style={{ padding: '16px 24px', fontSize: '13px', fontWeight: 600 }}>
                          {e.id.slice(0, 8)}
                        </td>
                        <td><StatusBadge status={e.status} /></td>
                        <td style={{ fontWeight: 600, color: 'var(--primary)' }}>
                          {cfg.model_name?.replace(/_/g, ' ') || 'Searching...'}
                        </td>
                        <td className="td-mono" style={{ color: 'var(--accent)', fontWeight: 700 }}>
                          {metrics.primary_metric != null ? metrics.primary_metric.toFixed(4) : '—'}
                        </td>
                        <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                          {metrics.train_time != null ? `${metrics.train_time.toFixed(2)}s` : '—'}
                        </td>
                        <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>{e.n_trials ?? '—'}</td>
                        <td>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <button
                              className="btn btn-outline"
                              style={{ padding: '6px 12px', fontSize: '11px' }}
                              onClick={() => navigate(`/optimization?exp=${e.id}`)}
                            >
                              <Icons.Zap size={14} />
                              Analysis
                            </button>
                            <button
                              className="btn btn-outline"
                              style={{ padding: '6px 12px', fontSize: '11px' }}
                              onClick={() => navigate(`/explainability`)}
                            >
                              <Icons.Search size={14} />
                              Explain
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
