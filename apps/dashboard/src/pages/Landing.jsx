import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Icons } from '../components/Icons';
import AgenticFlowDiagram from '../components/AgenticFlowDiagram';

export default function Landing() {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Icons.Network />,
      title: "Knowledge Graph Integration",
      desc: "Datasets are nodes in a dynamic similarity network. We route optimizations based on topological neighbors."
    },
    {
      icon: <Icons.Cpu />,
      title: "GNN Meta-Learner",
      desc: "Zero-shot pipeline recommendation powered by deep Graph Convolutional Networks trained on cross-dataset meta-data."
    },
    {
      icon: <Icons.Activity />,
      title: "Uncertainty Quantization",
      desc: "MC Dropout providing epistemic uncertainty scores. Know when the system is guessing vs. certain."
    },
    {
      icon: <Icons.Zap />,
      title: "Multi-Objective NSGA-II",
      desc: "Simultaneous optimization for Accuracy, Latency, and Carbon footprint on the Pareto front."
    },
    {
      icon: <Icons.Shield />,
      title: "Federated Learning",
      desc: "Privacy-preserving meta-knowledge aggregation across organizations without raw data sharing."
    },
    {
      icon: <Icons.Search />,
      title: "Deep Explainability",
      desc: "SHAP-based feature importance coupled with LLM-generated rationale for every recommendation."
    }
  ];

  return (
    <div className="animate-fade">
      <section className="hero" style={{ padding: '120px 0 60px', position: 'relative' }}>
        {/* Ambient glow behind hero */}
        <div style={{ position: 'absolute', top: -100, left: '50%', transform: 'translateX(-50%)', width: 600, height: 600, background: 'var(--gradient-primary)', opacity: 0.1, filter: 'blur(100px)', borderRadius: '50%', pointerEvents: 'none', zIndex: -1 }} />
        
        <div className="page-container">
          <span className="hero-tag" style={{ background: 'white', border: '1px solid var(--border)', boxShadow: 'var(--shadow-sm)' }}>
            <span className="text-gradient-primary">Project MetaLearnX System</span>
          </span>
          <h1 className="hero-title" style={{ fontSize: 72 }}>
            Intelligence That<br />
            <span className="text-gradient-secondary">Learns</span> From Intelligence.
          </h1>
          <p className="hero-subtitle" style={{ fontSize: 22, maxWidth: 680 }}>
            An autonomous multi-agent orchestration layer that uses meta-learning, 
            graph neural networks, and semantic analysis to synthesize optimal ML pipelines with zero human code.
          </p>
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
            <button 
              className="btn btn-primary btn-lg"
              style={{ background: 'var(--gradient-primary)', border: 'none', boxShadow: '0 10px 20px -5px rgba(255,0,128,0.4)', padding: '16px 40px', fontSize: 18 }}
              onClick={() => navigate('/runner')}
            >
              Initialize Workspace
              <Icons.ArrowRight />
            </button>
            <button 
              className="btn btn-outline btn-lg"
              style={{ padding: '16px 40px', fontSize: 18, background: 'white', border: '1px solid var(--border)' }}
              onClick={() => window.open('https://github.com', '_blank')}
            >
              View Documentation
              <Icons.Code />
            </button>
          </div>
        </div>
      </section>

      {/* --- Agentic Flow Diagram Section --- */}
      <section style={{ padding: '40px 0 80px' }}>
        <div className="page-container">
          <div style={{ textAlign: 'center', marginBottom: 40 }}>
            <h2 style={{ fontSize: 14, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: 8 }}>System Architecture Flow</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: 16 }}>Hover over the autonomous reasoning manifold below.</p>
          </div>
          <AgenticFlowDiagram />
        </div>
      </section>

      <section style={{ background: 'white', padding: '100px 0', borderTop: '1px solid var(--border)' }}>
        <div className="page-container">
          <div style={{ textAlign: 'center', marginBottom: 60 }}>
            <h2 style={{ fontSize: 32, fontWeight: 700, marginBottom: 16 }}>Research-Grade Capabilities</h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto' }}>
              Built for top-tier academic implementation, MetaLearnX combines industry-standard 
              AutoML with experimental state-of-the-art architectures.
            </p>
          </div>

          <div className="grid-features">
            {features.map((f, i) => (
              <div key={i} className="card feature-card">
                <div className="feature-icon">
                  {f.icon}
                </div>
                <h3 style={{ marginBottom: 12, fontSize: 18 }}>{f.title}</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '100px 0' }}>
        <div className="page-container">
          <div className="card" style={{ 
            background: 'var(--primary)', 
            color: 'white', 
            padding: 60, 
            borderRadius: 'var(--radius-xl)',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
          }}>
            <h2 style={{ fontSize: 40, fontWeight: 800, marginBottom: 20 }}>Ready to revolutionize your ML workflow?</h2>
            <p style={{ opacity: 0.8, fontSize: 18, maxWidth: 600, marginBottom: 40 }}>
              Upload your first dataset and experience the power of meta-learning driven pipeline synthesis.
            </p>
            <button 
              className="btn btn-primary btn-lg" 
              style={{ background: 'white', color: 'var(--primary)' }}
              onClick={() => navigate('/runner')}
            >
              Get Started Now
              <Icons.Zap />
            </button>
          </div>
        </div>
      </section>

      <footer style={{ padding: '60px 0', borderTop: '1px solid var(--border)', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
          &copy; 2026 MetaLearnX Research Lab. Distributed under MIT License.
        </p>
      </footer>
    </div>
  );
}
