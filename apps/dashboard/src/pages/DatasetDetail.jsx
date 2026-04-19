import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts';

const FMT = (v) => (typeof v === 'number' ? v.toFixed(4) : String(v));

export default function DatasetDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: ds, isLoading } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => api.getDataset(id),
    enabled: !!id,
  });

  const { data: similar } = useQuery({
    queryKey: ['similar', id],
    queryFn: () => api.getSimilar(id),
    enabled: !!id,
  });

  const [targetCol, setTargetCol] = useState('target');

  const runMutation = useMutation({
    mutationFn: () =>
      api.runExperiment({
        dataset_id: id,
        target_col: targetCol,
        task_type: ds?.task_type || 'classification',
        multi_objective: true,
      }),
    onSuccess: (data) => {
      toast.success('Optimization initiated');
      navigate(`/optimization?exp=${data.experiment_id}`);
    },
    onError: (err) => toast.error(err.response?.data?.detail || 'Execution failed'),
  });

  if (isLoading) return (
    <div className="loading-center">
      <div className="spinner" />
      <span>Calibrating meta-features...</span>
    </div>
  );

  if (!ds) return (
    <div className="page-container">
      <div className="empty-state">
        <Icons.Activity size={48} color="var(--danger)" />
        <div className="empty-title">Dataset Asset Undefined</div>
        <div className="empty-sub">The requested resource could not be located in the local meta-db.</div>
      </div>
    </div>
  );

  const profile = ds.meta_features || {};
  const alerts = ds.alerts || [];

  const radarData = [
    { metric: 'Complexity', value: (profile.complexity_score || 0) * 100 },
    { metric: 'Missing', value: (profile.missing_rate_mean || 0) * 100 },
    { metric: 'Imbalance', value: (1 - (profile.class_imbalance_ratio || 1)) * 100 },
    { metric: 'Correlation', value: (profile.corr_mean || 0) * 100 },
    { metric: 'Outliers', value: (profile.outlier_rate || 0) * 100 },
    { metric: 'High-Card', value: (profile.high_cardinality_ratio || 0) * 100 },
  ];

  const mfItems = [
    ['Analytical Task', ds.task_type],
    ['Sample Magnitude', ds.n_samples?.toLocaleString()],
    ['Feature Dimension', ds.n_features],
    ['Numeric Signal', profile.n_numeric],
    ['Categorical Signal', profile.n_categorical],
    ['Missing Propensity', `${((profile.missing_rate_mean || 0) * 100).toFixed(2)}%`],
    ['Class Skew', FMT(profile.class_imbalance_ratio)],
    ['Feature Sparsity', FMT(profile.sparsity)],
    ['Vector Complexity', FMT(profile.complexity_score)],
  ];

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
              <h1 className="page-title">{ds.name}</h1>
              <span className="badge badge-indigo">{ds.task_type}</span>
            </div>
            <p className="page-subtitle">
              Internal ID: <span className="td-mono">{ds.id.slice(0, 8)}...</span> &bull; 
              Source: <span style={{ fontWeight: 600 }}>{ds.source}</span>
            </p>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              className="btn btn-outline"
              onClick={() => navigate('/upload')}
            >
              <Icons.Upload />
              New Ingestion
            </button>
            <button
              className="btn btn-primary"
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
            >
              {runMutation.isPending ? 'Propagating...' : 'Initiate Optimization Pipeline'}
              <Icons.Zap />
            </button>
          </div>
        </div>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        {/* Quality Alerts */}
        {alerts.length > 0 && (
          <div style={{ marginBottom: 32 }}>
            {alerts.map((a, i) => (
              <div key={i} className="alert alert-info">
                <Icons.Search size={16} />
                <span style={{ fontSize: '14px', fontWeight: 500 }}>{a}</span>
              </div>
            ))}
          </div>
        )}

        {/* Description section from Stage 1 if exists */}
        {ds.description && (
          <div className="card" style={{ marginBottom: 32, borderLeft: '4px solid var(--accent)' }}>
            <h3 className="card-title">
              <Icons.Eye size={18} />
              Semantic Context
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: 15, fontStyle: 'italic', lineHeight: 1.7 }}>
              "{ds.description}"
            </p>
          </div>
        )}

        <div className="grid-2">
          {/* Intelligent Profiling */}
          <div className="card">
            <h3 className="card-title">
              <Icons.Box size={18} />
              Structural Meta-Features
            </h3>
            <div className="table-wrapper">
              <table>
                <tbody>
                  {mfItems.map(([label, value]) => (
                    <tr key={label}>
                      <td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</td>
                      <td className="td-mono" style={{ textAlign: 'right' }}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            {/* Complexity Radar */}
            <div className="card">
              <h3 className="card-title">
                <Icons.Activity size={18} />
                Profile Topology
              </h3>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                      name="Dataset"
                      dataKey="value"
                      stroke="var(--accent)"
                      fill="var(--accent)"
                      fillOpacity={0.15}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: '12px', 
                        border: '1px solid var(--border)',
                        boxShadow: 'var(--shadow-lg)'
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Similarity Module */}
            <div className="card">
              <h3 className="card-title">
                <Icons.Network size={18} />
                Similarity Neighbors
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {(similar?.similar_datasets || []).length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: 13, textAlign: 'center', padding: '20px 0' }}>
                    No neighbors identified in current manifold.
                  </p>
                ) : (
                  similar.similar_datasets.map((n) => (
                    <div key={n.neighbor_id} className="similarity-card" style={{ padding: '16px', background: 'var(--bg-base)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600, fontSize: 14 }}>{n.neighbor_name}</span>
                        <span style={{ color: 'var(--accent)', fontWeight: 700, fontFamily: 'monospace' }}>
                          {(n.similarity_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      {n.best_experiment && (
                        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)', display: 'flex', gap: 6 }}>
                          <Icons.Cpu size={14} />
                          Optimal Model: <span style={{ fontWeight: 600 }}>{n.best_experiment.model}</span>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
