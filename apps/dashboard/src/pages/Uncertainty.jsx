import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ErrorBar, Cell, ReferenceLine,
} from 'recharts';

const UNCERTAINTY_COLORS = {
  very_confident: '#10b981',   // emerald-500
  confident: '#4f46e5',        // indigo-600
  uncertain: '#f59e0b',        // amber-500
  very_uncertain: '#ef4444',   // red-500
};

export default function Uncertainty() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('dataset_id');
  const [nPasses, setNPasses] = useState(50);
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasetId || '');

  const { data: datasetsData } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.listDatasets,
  });
  const datasets = datasetsData?.datasets || [];

  const { data: result, isLoading, isError, refetch } = useQuery({
    queryKey: ['uncertainty', selectedDatasetId, nPasses],
    queryFn: () => api.getUncertainRecommend(selectedDatasetId, nPasses),
    enabled: !!selectedDatasetId,
  });

  const recommendations = result?.recommendations || [];
  const entropy = result?.predictive_entropy;
  const globalUncertainty = result?.global_uncertainty;
  const nPassesActual = result?.n_passes;

  const chartData = recommendations.map((r) => ({
    name: r.model_name.replace(/_/g, ' '),
    mean: r.mean_confidence,
    std: r.std,
    errorY: [r.mean_confidence - r.ci_95_low, r.ci_95_high - r.mean_confidence],
    label: r.uncertainty_label,
  }));

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Uncertainty Quantification</h1>
        <p className="page-subtitle">
          Bayesian approximation via MC Dropout. Visualizing the epistemic limits of the meta-learner manifold.
        </p>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        {/* Configuration */}
        <div className="card" style={{ marginBottom: 32 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px 180px', gap: 20, alignItems: 'flex-end' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Icons.Box size={14} />
                Target Dataset
              </label>
              <select
                className="form-select"
                style={{ padding: '12px 16px', fontWeight: 500 }}
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
              >
                <option value="">Select Research Asset...</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name} &bull; {d.n_samples?.toLocaleString()} samples</option>
                ))}
              </select>
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">MC Passes (T)</label>
              <input
                type="number" min={10} max={200} step={10}
                value={nPasses}
                onChange={(e) => setNPasses(Number(e.target.value))}
                className="form-input"
                style={{ height: 48, fontWeight: 600, textAlign: 'center' }}
              />
            </div>
            <button
              className="btn btn-primary btn-lg"
              style={{ height: 48, width: '100%' }}
              onClick={() => refetch()}
              disabled={!selectedDatasetId || isLoading}
            >
              {isLoading ? 'Estimating...' : 'Quantify Risk'}
              <Icons.Activity />
            </button>
          </div>
        </div>

        {/* Global Uncertainty Metrics */}
        {result && (
          <div className="metrics-grid" style={{ marginBottom: 32 }}>
            <div className="metric-card" style={{ borderLeft: `4px solid ${UNCERTAINTY_COLORS[globalUncertainty] || 'var(--border)'}` }}>
              <div className="metric-label">System Entropy</div>
              <div className="metric-value">{entropy?.toFixed(4) ?? '—'}</div>
              <div className="metric-sub">Pred. Distribution Spread</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Epistemic State</div>
              <div className="metric-value" style={{ 
                fontSize: 20, 
                color: UNCERTAINTY_COLORS[globalUncertainty] || 'var(--primary)',
                textTransform: 'capitalize'
              }}>
                {globalUncertainty?.replace(/_/g, ' ')}
              </div>
              <div className="metric-sub">Global Classification</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Operational Passes</div>
              <div className="metric-value">{nPassesActual || 0}</div>
              <div className="metric-sub">Monte Carlo Iterations</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Confidence Mean</div>
              <div className="metric-value">
                {(recommendations[0]?.mean_confidence * 100 || 0).toFixed(1)}%
              </div>
              <div className="metric-sub">Top Rank Estimator</div>
            </div>
          </div>
        )}

        {recommendations.length > 0 && (
          <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) 300px' }}>
            {/* Main Distribution Chart */}
            <div className="card">
              <h3 className="card-title">
                <Icons.BarChart size={18} />
                Confidence Distribution with 95% CI
              </h3>
              <div style={{ height: 320, width: '100%', marginTop: 24 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#64748b', fontSize: 11, fontWeight: 500 }}
                      angle={-20}
                      textAnchor="end"
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: '#f8fafc' }}
                      contentStyle={{
                        borderRadius: '12px', border: '1px solid var(--border)',
                        boxShadow: 'var(--shadow-lg)', fontSize: 12
                      }}
                      formatter={(v, name) => name === 'mean' ? [`${(v * 100).toFixed(2)}%`, 'Confidence'] : []}
                    />
                    <Bar dataKey="mean" radius={[4, 4, 0, 0]} barSize={40}>
                      {chartData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={UNCERTAINTY_COLORS[entry.label] || 'var(--accent)'}
                          fillOpacity={0.8}
                        />
                      ))}
                      <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="var(--primary)" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Side Research Note */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div className="card" style={{ background: 'var(--primary)', color: 'white' }}>
                <h3 className="card-title" style={{ color: 'white' }}>
                  <Icons.Shield size={18} color="white" />
                  Probabilistic Guard
                </h3>
                <p style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.8 }}>
                  Unlike deterministic AutoML, MetaLearnX utilizes <strong style={{color: 'white'}}>Dropout as a Bayesian Approximation</strong>. 
                  By performing $T$ stochastic forward passes, we recover the model's posterior distribution, 
                  shielding the research pipeline from over-confident but incorrect zero-shot predictions.
                </p>
              </div>

              <div className="card" style={{ background: '#fafafa', borderStyle: 'dashed' }}>
                <h3 className="card-title">Interpretation Key</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {Object.entries(UNCERTAINTY_COLORS).map(([label, color]) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ width: 10, height: 10, borderRadius: '50%', background: color }} />
                      <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                        {label.replace(/_/g, ' ')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Detailed Table */}
        {recommendations.length > 0 && (
          <div className="card" style={{ marginTop: 32, padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '24px 28px', borderBottom: '1px solid var(--border)' }}>
               <h3 className="card-title" style={{ margin: 0 }}>Quantized Confidence Intervals</h3>
            </div>
            <div className="table-wrapper">
              <table style={{ border: 'none' }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '16px 24px' }}>Model Topology</th>
                    <th>Mean Score</th>
                    <th>Std. Deviation</th>
                    <th>95% Interval (Low)</th>
                    <th>95% Interval (High)</th>
                    <th>Risk State</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.map((r) => (
                    <tr key={r.model_name}>
                      <td style={{ padding: '16px 24px', fontWeight: 600, color: 'var(--primary)' }}>{r.model_name}</td>
                      <td className="td-mono" style={{ color: 'var(--accent)', fontWeight: 700 }}>
                        {(r.mean_confidence * 100).toFixed(2)}%
                      </td>
                      <td style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>&plusmn;{(r.std * 100).toFixed(2)}%</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{(r.ci_95_low * 100).toFixed(2)}%</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{(r.ci_95_high * 100).toFixed(2)}%</td>
                      <td>
                        <span className="badge" style={{ 
                          backgroundColor: (UNCERTAINTY_COLORS[r.uncertainty_label] || '#94a3b8') + '15',
                          color: UNCERTAINTY_COLORS[r.uncertainty_label] || '#94a3b8',
                          textTransform: 'uppercase',
                          fontSize: '10px'
                        }}>
                          {r.uncertainty_label?.replace(/_/g, ' ')}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!selectedDatasetId && (
          <div className="empty-state" style={{ padding: '100px 0' }}>
            <Icons.Activity size={64} style={{ color: 'var(--border)', marginBottom: 24 }} />
            <h2 className="empty-title">Uncertainty Workspace Null</h2>
            <p className="empty-sub">Select an ingested asset above to begin stochastic MC Dropout quantification.</p>
          </div>
        )}
      </div>
    </div>
  );
}
