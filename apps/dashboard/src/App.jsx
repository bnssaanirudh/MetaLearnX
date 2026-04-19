import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

import Landing from './pages/Landing';
import PipelineRunner from './pages/PipelineRunner';
import DatasetDetail from './pages/DatasetDetail';
import Recommendations from './pages/Recommendations';
import Optimization from './pages/Optimization';
import Explainability from './pages/Explainability';
import History from './pages/History';
import Benchmark from './pages/Benchmark';
import KnowledgeGraph from './pages/KnowledgeGraph';
import Uncertainty from './pages/Uncertainty';
import FederatedLearning from './pages/FederatedLearning';
import { api } from './api/client';
import { Icons } from './components/Icons';
import ErrorBoundary from './components/ErrorBoundary';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 10_000 } },
});

const NAV = [
  { to: '/', icon: <Icons.Home />, label: 'Overview', exact: true },
  { to: '/runner', icon: <Icons.Activity />, label: 'Live Execution' },
  { to: '/history', icon: <Icons.Database />, label: 'Experiments', section: 'Meta-Learning' },
  { to: '/recommendations', icon: <Icons.Target />, label: 'Recommendations' },
  { to: '/optimization', icon: <Icons.Zap />, label: 'Optimization' },
  { to: '/explainability', icon: <Icons.Search />, label: 'Explainability' },
  { to: '/uncertainty', icon: <Icons.Activity />, label: 'Uncertainty' },
  { to: '/knowledge-graph', icon: <Icons.Globe />, label: 'Knowledge Graph', section: 'Advanced Research' },
  { to: '/federated', icon: <Icons.Share />, label: 'Federated Learning' },
  { to: '/benchmark', icon: <Icons.BarChart />, label: 'Benchmark' },
];

function Sidebar() {
  const { data: stats } = useQuery({
    queryKey: ['meta-db-stats'],
    queryFn: api.getMetaDbStats,
    refetchInterval: 30_000,
  });

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-text">MetaLearnX</div>
        <div className="logo-sub">Autonomous Research Lab</div>
      </div>

      <nav className="sidebar-nav">
        {NAV.map((item, idx) => (
          <div key={idx}>
            {item.section && (
              <div className="nav-section-label">{item.section}</div>
            )}
            <NavLink
              to={item.to}
              end={item.exact}
              className={({ isActive }) =>
                `nav-item${isActive ? ' active' : ''}`
              }
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="nav-section-label" style={{ padding: '0 0 12px 0' }}>Knowledge Base</div>
        <div className="db-stat" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Datasets</span>
          <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{stats?.n_datasets ?? '—'}</span>
        </div>
        <div className="db-stat" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Knowledge Nodes</span>
          <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{stats?.n_experiments ?? '—'}</span>
        </div>
      </div>
    </aside>
  );
}

const AppLayout = ({ children }) => {
  const location = useLocation();
  const isLandingPage = location.pathname === '/';

  return (
    <div className="app-layout">
      {!isLandingPage && <Sidebar />}
      <main className="main-content" style={isLandingPage ? { marginLeft: 0, padding: 0 } : {}}>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </main>
    </div>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppLayout>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/runner" element={<PipelineRunner />} />
            <Route path="/dataset/:id" element={<DatasetDetail />} />
            <Route path="/recommendations" element={<Recommendations />} />
            <Route path="/optimization" element={<Optimization />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/history" element={<History />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
            <Route path="/uncertainty" element={<Uncertainty />} />
            <Route path="/federated" element={<FederatedLearning />} />
          </Routes>
        </AppLayout>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#ffffff',
              color: '#0f172a',
              border: '1px solid #e2e8f0',
              borderRadius: '10px',
              fontSize: '14px',
              fontWeight: '500',
              boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
            },
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
