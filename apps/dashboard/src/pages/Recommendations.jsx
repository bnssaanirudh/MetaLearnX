import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const MODEL_COLORS = {
  random_forest: '#4f46e5', // indigo-600
  xgboost: '#2563eb',       // blue-600
  lightgbm: '#0891b2',      // cyan-600
  catboost: '#7c3aed',      // violet-600
  logistic_regression: '#475569', // slate-600
  svm: '#0f172a',           // slate-900
  mlp: '#10b981',           // emerald-500
  tabpfn: '#d946ef',        // fuchsia-500
};

function RankCard({ rec, idx }) {
  const rankClass = idx < 3 ? `rank-${idx + 1}` : 'rank-n';
  const conf = (rec.confidence * 100).toFixed(1);

  return (
    <div className="rank-card animate-fade" style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)' }}>
      <div className={`rank-number ${rankClass}`} style={{ 
        width: 32, 
        height: 32, 
        borderRadius: '50%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        fontSize: 12,
        fontWeight: 800
      }}>
        {idx + 1}
      </div>
      <div className="rank-info" style={{ flex: 1, marginLeft: 16 }}>
        <div className="rank-name" style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
          {rec.model_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
        </div>
        <div className="rank-rationale" style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
          {rec.rationale || `Source: ${rec.source}`}
        </div>
      </div>
      <div className="confidence-bar-container" style={{ width: 140 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Confidence</span>
          <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent)' }}>{conf}%</span>
        </div>
        <div className="progress-bar" style={{ height: 4, background: 'var(--bg-base)' }}>
          <div className="progress-fill" style={{ width: `${conf}%`, background: 'var(--accent)' }} />
        </div>
      </div>
    </div>
  );
}

export default function Recommendations() {
  const { data: dsList } = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets });
  const datasets = dsList?.datasets || [];

  const [selectedDs, setSelectedDs] = useState('');
  const activeId = selectedDs || datasets[0]?.id;
  const activeDs = datasets.find(d => d.id === activeId);

  const { data: recs, isLoading } = useQuery({
    queryKey: ['recommendations', activeId],
    queryFn: () => api.getRecommendations(activeId, activeDs?.task_type || 'classification'),
    enabled: !!activeId,
  });

  const knnRecs = recs?.knn_recommendations || [];
  const neuralRecs = recs?.neural_recommendations || [];
  const [tab, setTab] = useState('knn');
  const shown = tab === 'knn' ? knnRecs : neuralRecs;

  const barData = shown.map(r => ({
    name: r.model_name.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase()),
    confidence: parseFloat((r.confidence * 100).toFixed(1)),
    model: r.model_name,
  }));

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Zero-Shot Intelligence</h1>
        <p className="page-subtitle">
          Advanced meta-learning engines predicting optimal pipeline candidates prior to empirical search.
        </p>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        <div className="card" style={{ marginBottom: 32 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Icons.Box size={14} />
              Active Research Context
            </label>
            <select
              className="form-select"
              style={{ padding: '12px 16px', fontWeight: 500 }}
              value={activeId || ''}
              onChange={(e) => setSelectedDs(e.target.value)}
            >
              {datasets.map(d => (
                <option key={d.id} value={d.id}>
                  {d.name} [{d.task_type.toUpperCase()}] &bull; {d.n_samples?.toLocaleString()} samples
                </option>
              ))}
            </select>
          </div>
        </div>

        {isLoading ? (
          <div className="loading-center" style={{ height: 300 }}>
            <div className="spinner" />
            <span style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-secondary)' }}>Querying Meta-Knowledge Base...</span>
          </div>
        ) : (
          <div className="grid-2">
            <div>
              <div className="tabs" style={{ marginBottom: 24 }}>
                <button 
                  className={`tab ${tab === 'knn' ? 'active' : ''}`} 
                  onClick={() => setTab('knn')}
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  <Icons.Network size={14} />
                  kNN Manifold
                </button>
                <button 
                  className={`tab ${tab === 'neural' ? 'active' : ''}`} 
                  onClick={() => setTab('neural')}
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  <Icons.Cpu size={14} />
                  Neural Synthesis
                </button>
              </div>

              <div style={{ marginBottom: 20, fontSize: 13, color: 'var(--text-secondary)', padding: '0 4px', lineHeight: 1.6 }}>
                {tab === 'knn'
                  ? 'Instance-based learning utilizing topological dataset similarity in the joint meta-feature space.'
                  : 'Deep probabilistic reasoning via MLP architecture trained on historical pipeline performance distributions.'
                }
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {shown.length === 0 ? (
                  <div className="empty-state">
                    <Icons.Target size={40} color="var(--border-bright)" />
                    <div className="empty-title">Zero-Shot Output Null</div>
                    <div className="empty-sub">Insufficient meta-data to generate high-confidence rankings.</div>
                  </div>
                ) : (
                  shown.map((rec, i) => <RankCard key={rec.model_name} rec={rec} idx={i} />)
                )}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div className="card">
                <h3 className="card-title">
                  <Icons.BarChart size={18} />
                  Confidence Projections
                </h3>
                <div style={{ height: 280, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={barData} layout="vertical" margin={{ left: 20, right: 30 }}>
                      <XAxis type="number" domain={[0, 100]} hide />
                      <YAxis 
                        type="category" 
                        dataKey="name" 
                        tick={{ fill: '#64748b', fontSize: 12, fontWeight: 500 }} 
                        width={130} 
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: 'var(--bg-base)' }}
                        contentStyle={{
                          borderRadius: '12px', border: '1px solid var(--border)',
                          boxShadow: 'var(--shadow-lg)', fontSize: 12,
                        }}
                        formatter={(v) => [`${v}%`, 'Probability']}
                      />
                      <Bar dataKey="confidence" radius={[0, 4, 4, 0]} barSize={20}>
                        {barData.map((d, i) => (
                          <Cell key={i} fill={MODEL_COLORS[d.model] || 'var(--accent)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card" style={{ background: '#fafafa', borderStyle: 'dashed' }}>
                <h3 className="card-title">Research Methodology</h3>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <p style={{ marginBottom: 16 }}>
                    <strong style={{ color: 'var(--primary)' }}>Topological Search:</strong> The system maps the target dataset into a standardized latent space using 33 statistical meta-features and domain-specific embeddings.
                  </p>
                  <p>
                    <strong style={{ color: 'var(--primary)' }}>Warm-Start Injection:</strong> These rankings are injected into the Multi-Objective Optimizer to prioritize high-potential regions of the search space, accelerating convergence.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
