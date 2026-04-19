import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import toast from 'react-hot-toast';

const MODEL_COLORS = {
  random_forest: '#4f46e5', // indigo-600
  xgboost: '#2563eb',       // blue-600
  lightgbm: '#0891b2',      // cyan-600
  catboost: '#7c3aed',      // violet-600
  logistic_regression: '#475569', // slate-600
  svm: '#0f172a',           // slate-900
  mlp: '#10b981',           // emerald-500
  unknown: '#94a3b8',
};

export default function KnowledgeGraph() {
  const { data: graph, isLoading, refetch } = useQuery({
    queryKey: ['knowledge-graph'],
    queryFn: api.getKnowledgeGraph,
  });

  const rebuildMutation = useMutation({
    mutationFn: api.rebuildKnowledgeGraph,
    onSuccess: () => {
      toast.success('Graph topology reconstruction initiated.');
      setTimeout(() => refetch(), 5000);
    },
    onError: () => toast.error('Reconstruction sequence failed'),
  });

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  const adjacency = {};
  edges.forEach((e) => {
    adjacency[e.source] = adjacency[e.source] || [];
    adjacency[e.target] = adjacency[e.target] || [];
    adjacency[e.source].push({ id: e.target, weight: e.weight });
    adjacency[e.target].push({ id: e.source, weight: e.weight });
  });

  const layoutNodes = nodes.map((n, i) => {
    const angle = (i / nodes.length) * 2 * Math.PI;
    const r = 240;
    return {
      ...n,
      x: Math.cos(angle) * r + 380,
      y: Math.sin(angle) * r + 300,
    };
  });
  const nodeMap = Object.fromEntries(layoutNodes.map((n) => [n.id, n]));

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
              <h1 className="page-title">Relational Knowledge Topology</h1>
              {graph?.pyg_available && (
                <span className="badge badge-success" style={{ height: 'fit-content' }}>GNN Active</span>
              )}
            </div>
            <p className="page-subtitle">
              Interactive dataset similarity manifold. Nodes represent project assets; edges represent cosine proximity.
            </p>
          </div>
          <button
            className="btn btn-primary"
            onClick={() => rebuildMutation.mutate()}
            disabled={rebuildMutation.isPending}
          >
            {rebuildMutation.isPending ? 'Propagating Graph...' : 'Reconstruct Topology'}
            <Icons.Activity />
          </button>
        </div>
      </header>

      {isLoading ? (
        <div className="loading-center" style={{ height: 400 }}>
          <div className="spinner" />
          <span style={{ fontSize: 14 }}>Mapping Topological Manifold...</span>
        </div>
      ) : (
        <div className="page-body" style={{ padding: 0 }}>
          <div className="metrics-grid" style={{ marginBottom: 32 }}>
            <div className="metric-card">
              <div className="metric-label">Dataset Nodes</div>
              <div className="metric-value">{graph?.n_nodes ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Similarity Edges</div>
              <div className="metric-value">{graph?.n_edges ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Neural Engine</div>
              <div className="metric-value" style={{ fontSize: 18 }}>{graph?.pyg_available ? 'Torch-GNN' : 'NetworkX'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Weight Threshold</div>
              <div className="metric-value">0.50 cos</div>
            </div>
          </div>

          <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) 340px' }}>
            {/* Graph Visualization */}
            <div className="card" style={{ overflow: 'hidden', padding: 40 }}>
              <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
                <svg width="760" height="600" style={{ maxWidth: '100%', height: 'auto' }}>
                  {edges.map((e, i) => {
                    const src = nodeMap[e.source];
                    const tgt = nodeMap[e.target];
                    if (!src || !tgt) return null;
                    return (
                      <line
                        key={i}
                        x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
                        stroke="#e2e8f0"
                        strokeWidth={Math.max(1, e.weight * 5)}
                        strokeOpacity={0.6}
                      />
                    );
                  })}
                  {layoutNodes.map((n) => {
                    const color = MODEL_COLORS[n.best_model] || MODEL_COLORS.unknown;
                    return (
                      <g key={n.id}>
                        <circle
                          cx={n.x} cy={n.y} r={24}
                          fill="white"
                          stroke={color}
                          strokeWidth={3}
                          style={{ filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.05))' }}
                        />
                        <text
                          x={n.x} y={n.y + 4}
                          textAnchor="middle"
                          fill="var(--text-primary)"
                          fontSize="10"
                          fontWeight="700"
                        >
                          {n.name?.length > 8 ? n.name.slice(0, 7) + '…' : n.name}
                        </text>
                        {n.best_metric > 0 && (
                          <text
                            x={n.x} y={n.y + 42}
                            textAnchor="middle"
                            fill="var(--text-muted)"
                            fontSize="9"
                            fontWeight="600"
                          >
                            {n.best_metric.toFixed(3)}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div style={{ marginTop: 40, borderTop: '1px solid var(--border)', paddingTop: 24 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16 }}>
                  {Object.entries(MODEL_COLORS).filter(([k]) => k !== 'unknown').map(([m, c]) => (
                    <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ width: 10, height: 10, borderRadius: '50%', background: c }} />
                      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                        {m.replace(/_/g, ' ')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Side Info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div className="card">
                <h3 className="card-title">
                  <Icons.Network size={18} />
                  Graph Reasoning
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.7 }}>
                  MetaLearnX treats the entire database as a <strong style={{color: 'var(--primary)'}}>Knowledge Graph</strong>. 
                  When a new dataset enters the system, our GNN (Graph Neural Network) performs topological message passing 
                  to predict optimal pipelines relative to its neighbors.
                </p>
              </div>

              <div className="card" style={{ background: '#f8fafc', borderStyle: 'dashed' }}>
                <h3 className="card-title">Node Attributes</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {nodes.slice(0, 8).map(n => (
                    <div key={n.id} style={{ padding: '10px 12px', background: 'white', borderRadius: 'var(--radius)', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, fontSize: 12 }}>{n.name}</span>
                      <span style={{ fontSize: 10, fontWeight: 800, color: 'var(--accent)' }}>{(adjacency[n.id] || []).length} deg</span>
                    </div>
                  ))}
                  {nodes.length > 8 && (
                    <p style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>+ {nodes.length - 8} more nodes</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Tabular Backup */}
          <div className="card" style={{ marginTop: 32, padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '24px 28px', borderBottom: '1px solid var(--border)' }}>
               <h3 className="card-title" style={{ margin: 0 }}>Inventory of Graph Entities</h3>
            </div>
            <div className="table-wrapper">
              <table style={{ border: 'none' }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '16px 24px' }}>Entity</th>
                    <th>Dominant Architecture</th>
                    <th>Optimal Signal</th>
                    <th>Analysis Type</th>
                    <th>Centrality</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map(n => (
                    <tr key={n.id}>
                      <td style={{ padding: '16px 24px', fontWeight: 600, color: 'var(--primary)' }}>{n.name}</td>
                      <td>
                        <span className="badge" style={{ backgroundColor: (MODEL_COLORS[n.best_model] || '#64748b') + '15', color: MODEL_COLORS[n.best_model] || '#64748b' }}>
                          {n.best_model || 'Undetermined'}
                        </span>
                      </td>
                      <td className="td-mono" style={{ fontWeight: 700 }}>{n.best_metric > 0 ? n.best_metric.toFixed(4) : '—'}</td>
                      <td><span className="badge badge-slate">{n.task_type}</span></td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{(adjacency[n.id] || []).length} connections</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
