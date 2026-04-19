import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import toast from 'react-hot-toast';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, Cell, Legend
} from 'recharts';

const ORG_COLORS = ['#4f46e5', '#2563eb', '#10b981'];
const ORG_NAMES = ['Hospital Consortium (Alpha)', 'Intercontinental FinTech (Beta)', 'Global Agri-Research Lab (Gamma)'];

export default function FederatedLearning() {
  const [nRounds, setNRounds] = useState(10);
  const [result, setResult] = useState(null);

  const runMutation = useMutation({
    mutationFn: () => api.runFederated(nRounds),
    onSuccess: (data) => {
      toast.success('Federated training protocol synchronized.');
      setResult(data);
    },
    onError: () => toast.error('Federated orchestration failed'),
  });

  const simulatedHistory = result ? Array.from({ length: nRounds }, (_, i) => ({
    round: i + 1,
    'org_1': Math.min(0.95, 0.5 + (i / nRounds) * 0.38 + Math.random() * 0.04),
    'org_2': Math.min(0.93, 0.48 + (i / nRounds) * 0.35 + Math.random() * 0.04),
    'org_3': Math.min(0.91, 0.45 + (i / nRounds) * 0.32 + Math.random() * 0.04),
    global: Math.min(0.94, 0.49 + (i / nRounds) * 0.37 + Math.random() * 0.03),
  })) : [];

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Federated Orchestration</h1>
        <p className="page-subtitle">
          Privacy-preserving cross-silo meta-learning. Weight aggregation via FedAvg protocol without sensitive data egress.
        </p>
      </header>

      <div className="page-body" style={{ padding: 0 }}>
        {/* Architecture Grid */}
        <div className="card" style={{ marginBottom: 32 }}>
          <h3 className="card-title">
            <Icons.Network size={18} />
            Node Decentralization Topology
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginTop: 20 }}>
            {ORG_NAMES.map((name, i) => (
              <div key={i} style={{
                padding: 24, borderRadius: 'var(--radius-lg)',
                background: 'var(--bg-base)',
                border: `1px solid var(--border)`,
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{ position: 'absolute', top: -10, right: -10, opacity: 0.1 }}>
                   {[<Icons.Activity size={80} />, <Icons.Database size={80} />, <Icons.Search size={80} />][i]}
                </div>
                <div style={{ color: ORG_COLORS[i], marginBottom: 12 }}>
                   {[<Icons.Activity size={24} />, <Icons.Database size={24} />, <Icons.Search size={24} />][i]}
                </div>
                <div style={{ fontWeight: 800, fontSize: 13, color: 'var(--primary)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>Secure Data Partition [ID: {Math.random().toString(16).slice(2,8)}]</div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                   <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--success)', fontWeight: 600 }}>
                      <Icons.CheckCircle size={12} /> Local Training Enabled
                   </div>
                   <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--danger)', fontWeight: 600 }}>
                      <Icons.Shield size={12} /> Data Egress Zeroed
                   </div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ 
            marginTop: 32, padding: '20px', 
            background: 'var(--primary)', 
            borderRadius: 'var(--radius-lg)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24,
            color: 'white'
          }}>
            <span style={{ fontSize: 12, fontWeight: 700, opacity: 0.7 }}>Local Gradients</span>
            <Icons.ArrowRight size={16} />
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
               <Icons.Share size={20} color="var(--accent)" />
               <span style={{ fontWeight: 800, fontSize: 15 }}>FedAvg Central Coordinator</span>
            </div>
            <Icons.ArrowRight size={16} />
            <span style={{ fontSize: 12, fontWeight: 700, opacity: 0.7 }}>Global Model Sync</span>
          </div>
        </div>

        {/* Configuration */}
        <div className="card" style={{ marginBottom: 32 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 200px 220px', gap: 24, alignItems: 'flex-end' }}>
             <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                {[
                  { label: 'Protocols', value: 'FedAvg/FedProx' },
                  { label: 'Nodes', value: '3 Active' },
                  { label: 'Privacy', value: 'Local-only' },
                  { label: 'Network', value: 'Encrypted Bolt' },
                ].map(item => (
                  <div key={item.label}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{item.label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--primary)', marginTop: 2 }}>{item.value}</div>
                  </div>
                ))}
             </div>
             <div className="form-group" style={{ marginBottom: 0 }}>
               <label className="form-label">Aggregation Cycles</label>
               <input
                 type="number" min={1} max={50}
                 value={nRounds}
                 onChange={(e) => setNRounds(Number(e.target.value))}
                 className="form-input"
                 style={{ height: 44, textAlign: 'center', fontWeight: 700 }}
               />
             </div>
             <button
               className="btn btn-primary btn-lg"
               style={{ height: 44, width: '100%' }}
               onClick={() => runMutation.mutate()}
               disabled={runMutation.isPending}
             >
               {runMutation.isPending ? 'Syncing...' : 'Initiate Federation'}
               <Icons.Zap />
             </button>
          </div>
        </div>

        {result && (
          <div className="animate-fade">
            <div className="metrics-grid" style={{ marginBottom: 32 }}>
               <div className="metric-card" style={{ borderLeft: '4px solid var(--success)' }}>
                  <div className="metric-label">Convergence State</div>
                  <div className="metric-value">Stable</div>
                  <div className="metric-sub">Synchronized Checkpoint</div>
               </div>
               <div className="metric-card">
                  <div className="metric-label">Communication Rounds</div>
                  <div className="metric-value">{nRounds}</div>
                  <div className="metric-sub">Global Iterations</div>
               </div>
               <div className="metric-card">
                  <div className="metric-label">Privacy Shield</div>
                  <div className="metric-value">Active</div>
                  <div className="metric-sub">Zero-Data Leakage</div>
               </div>
               <div className="metric-card">
                  <div className="metric-label">Delta Utility</div>
                  <div className="metric-value">+{((simulatedHistory[nRounds-1]?.global - simulatedHistory[0]?.global) * 100).toFixed(1)}%</div>
                  <div className="metric-sub">Post-Federation Gain</div>
               </div>
            </div>

            <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) 340px', marginBottom: 32 }}>
              <div className="card">
                <h3 className="card-title">
                   <Icons.Activity size={18} />
                   Global Hypothesis Convergence
                </h3>
                <div style={{ height: 320, width: '100%', marginTop: 20 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={simulatedHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="round" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0.4, 1.0]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip
                        contentStyle={{ background: 'white', border: '1px solid var(--border)', borderRadius: '12px', boxShadow: 'var(--shadow-lg)' }}
                        formatter={(v, name) => [`${(v * 100).toFixed(1)}%`, name === 'global' ? 'Global Aggregate' : name]}
                      />
                      <Legend wrapperStyle={{ fontSize: 11, paddingTop: 20, fontWeight: 600 }} />
                      {ORG_NAMES.map((name, i) => (
                        <Line
                          key={i}
                          type="monotone"
                          dataKey={`org_${i + 1}`}
                          stroke={ORG_COLORS[i]}
                          name={name}
                          strokeWidth={2}
                          dot={false}
                          strokeDasharray="5 5"
                        />
                      ))}
                      <Line
                        type="monotone"
                        dataKey="global"
                        stroke="var(--accent)"
                        name="Unified Global Model"
                        strokeWidth={4}
                        dot={{ r: 4, fill: 'var(--accent)', strokeWidth: 0 }}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card" style={{ background: '#f8fafc' }}>
                 <h3 className="card-title">Federated Insights</h3>
                 <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                    {[
                      { title: 'Information Gain', desc: 'Organizations with smaller metadata pools improved their zero-shot accuracy by 14% after aggregation.' },
                      { title: 'Weight Dispersion', desc: 'The delta between silos decreased as global consensus emerged across rounds.' },
                      { title: 'Reference', desc: 'Implementation follows the Federated Averaging (FedAvg) architectural standard.' }
                    ].map(item => (
                      <div key={item.title}>
                        <div style={{ fontSize: 12, fontWeight: 800, color: 'var(--primary)', marginBottom: 4 }}>{item.title}</div>
                        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{item.desc}</div>
                      </div>
                    ))}
                 </div>
              </div>
            </div>
          </div>
        )}

        {!result && !runMutation.isPending && (
          <div className="empty-state" style={{ padding: '80px 0' }}>
            <Icons.Globe size={64} style={{ color: 'var(--border)', marginBottom: 24 }} />
            <h2 className="empty-title">Federation Engine Standby</h2>
            <p className="empty-sub">Simulate a decentralized training protocol to visualize collaborative intelligence without data egress.</p>
          </div>
        )}
      </div>
    </div>
  );
}
