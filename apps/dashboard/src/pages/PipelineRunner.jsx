import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { api } from '../api/client';
import { Icons } from '../components/Icons';
import AgenticFlowDiagram from '../components/AgenticFlowDiagram';

export default function PipelineRunner() {
  const [file, setFile] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [description, setDescription] = useState('');
  
  const [activeExpId, setActiveExpId] = useState(null);
  
  const navigate = useNavigate();
  const qc = useQueryClient();

  // --- File Upload ---
  const uploadMutation = useMutation({
    mutationFn: () => api.uploadDataset(file, targetCol, taskType, description),
    onSuccess: (data) => {
      toast.success(`Success: ${data.name} initialized in research DB.`);
      qc.invalidateQueries(['datasets']);
      
      // We don't have an experiment ID yet from the upload response directly
      // but the backend does create one eventually or we can manually trigger it.
      // Actually, wait: /api/experiments/run currently takes dataset_id.
      // So we need to chain the run command.
      runExpMutation.mutate(data.dataset_id);
    },
    onError: (err) => {
      toast.error(err.response?.data?.detail || 'System error during upload');
    },
  });

  const runExpMutation = useMutation({
    mutationFn: (datasetId) => api.runExperiment({ dataset_id: datasetId, target_col: targetCol, task_type: taskType }),
    onSuccess: (data) => {
      toast.success('Agentic swarm deployed.');
      setActiveExpId(data.experiment_id);
    },
    onError: (err) => {
      toast.error('Failed to dispatch orchestrator.');
    }
  });

  const onDrop = useCallback((files) => {
    if (files[0]) {
      setFile(files[0]);
      if (!targetCol) setTargetCol('target');
      setActiveExpId(null); // reset if uploading new
    }
  }, [targetCol]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  // --- Real-time Polling ---
  const { data: expData } = useQuery({
    queryKey: ['exp-results', activeExpId],
    queryFn: () => api.getExpResults(activeExpId),
    enabled: !!activeExpId,
    refetchInterval: (data) => (data?.status === 'running' || data?.status === 'pending') ? 1500 : false,
  });

  // --- Keyboard Shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (file && targetCol && !uploadMutation.isPending && !runExpMutation.isPending && !activeExpId) {
          uploadMutation.mutate();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [file, targetCol, uploadMutation, runExpMutation, activeExpId]);

  // --- Derive Active State ---
  let activeState = 'idle';
  let logs = [];
  let bestConfig = {};
  let bestMetrics = {};

  if (uploadMutation.isPending) {
    activeState = 'profiling';
  } else if (runExpMutation.isPending) {
    activeState = 'profiling';
  } else if (expData) {
    logs = (expData.reasoning_logs || []).slice(-150); // DOM Limit to prevent React reconciliation bloat
    bestConfig = expData.pipeline_config || {};
    bestMetrics = expData.metrics || {};
    
    if (expData.status === 'pending') {
      activeState = 'profiling';
    } else if (expData.status === 'running') {
      if (logs.length === 0) {
        activeState = 'architecting';
      } else {
        const lastLog = logs[logs.length - 1];
        if (lastLog.agent === 'Architect') {
          activeState = 'optimizing';
        } else if (lastLog.agent === 'Critic') {
          // It flashes critic, then goes back to optimizing.
          // For visualization, if the last thing was a Critic pivot, show Critic.
          activeState = 'critic';
        } else {
          activeState = 'optimizing';
        }
      }
    } else if (expData.status === 'done') {
      activeState = 'done';
    }
  }

  // --- Sub-components rendering ---

  const renderUploadForm = () => (
    <div className="card animate-fade" style={{ display: 'flex', flexDirection: 'column', gap: 24, height: '100%' }}>
      <h3 className="card-title" style={{ marginBottom: 0 }}>
        <Icons.Upload size={20} />
        Initialize Workspace
      </h3>
      
      <div
        {...getRootProps()}
        style={{
          padding: '40px 20px',
          border: '2px dashed var(--border)',
          borderRadius: 'var(--radius-lg)',
          background: isDragActive ? 'var(--accent-light)' : 'var(--bg-base)',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'var(--transition)',
        }}
      >
        <input {...getInputProps()} />
        {file ? (
          <div>
            <Icons.CheckCircle size={32} color="var(--success)" style={{ margin: '0 auto 10px' }} />
            <div style={{ fontWeight: 700, color: 'var(--primary)' }}>{file.name}</div>
          </div>
        ) : (
           <div style={{ color: 'var(--text-secondary)' }}>
             <Icons.Upload size={32} color="var(--accent)" style={{ margin: '0 auto 10px' }} />
             Drag & Drop CSV Dataset
           </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label" style={{ fontSize: 11 }}>Target Variable</label>
          <input className="form-input" value={targetCol} onChange={(e) => setTargetCol(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label" style={{ fontSize: 11 }}>Task Type</label>
          <select className="form-select" value={taskType} onChange={(e) => setTaskType(e.target.value)}>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
      </div>
      
      <div className="form-group" style={{ marginBottom: 0 }}>
        <label className="form-label" style={{ fontSize: 11 }}>Semantic Context</label>
        <textarea className="form-input" style={{ minHeight: '60px' }} value={description} onChange={(e) => setDescription(e.target.value)} />
      </div>

      <button
        className="btn btn-primary btn-lg"
        style={{ width: '100%', marginTop: 'auto', background: 'var(--gradient-primary)', border: 'none' }}
        onClick={() => uploadMutation.mutate()}
        disabled={!file || !targetCol || uploadMutation.isPending || runExpMutation.isPending}
      >
        {(uploadMutation.isPending || runExpMutation.isPending) ? 'Igniting Agents...' : 'Execute Live Pipeline'}
        <Icons.Activity />
      </button>
    </div>
  );

  const renderLiveStatus = () => (
    <div className="card animate-fade" style={{ display: 'flex', flexDirection: 'column', gap: 20, height: '100%', borderLeft: '4px solid var(--accent)' }}>
      <h3 className="card-title" style={{ marginBottom: 0 }}>
        {activeState === 'done' ? <><Icons.CheckCircle color="var(--success)" /> Pipeline Finalized</> : <><div className="spinner" style={{ width: 18, height: 18 }} /> Orchestrator Active</>}
      </h3>
      
      {/* Real-time Logs Console */}
      <div style={{ flex: 1, background: '#0f172a', borderRadius: 'var(--radius)', padding: 16, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {logs.length === 0 && activeState !== 'done' && (
          <div style={{ color: '#00DFD8', fontSize: 13, fontFamily: 'monospace' }}>&gt; Extracting semantic dataset meta-features...</div>
        )}
        {logs.map((log, idx) => (
          <div key={idx} className="animate-fade" style={{ borderLeft: `2px solid ${log.agent === 'Architect' ? '#FF0080' : '#F9CB28'}`, paddingLeft: 10 }}>
            <div style={{ fontSize: 10, color: '#94a3b8', fontFamily: 'monospace', marginBottom: 4 }}>
              [{log.agent.toUpperCase()}][ROUND {log.iteration + 1}]
            </div>
            <div style={{ fontSize: 13, color: '#f8fafc', lineHeight: 1.5 }}>
              {log.reasoning}
            </div>
            {log.proposed_config && (
              <div style={{ fontSize: 11, color: '#00DFD8', marginTop: 4, fontFamily: 'monospace' }}>
                &gt; DEPLOYING BASE: {log.proposed_config.model_name.toUpperCase()}
              </div>
            )}
            {log.issue && (
              <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4, fontFamily: 'monospace' }}>
                &gt; ISSUE: {log.issue} <br/>
                <span style={{ color: '#10b981' }}>&gt; PIVOT: {log.refinement}</span>
              </div>
            )}
          </div>
        ))}
        {activeState === 'optimizing' && (
           <div className="animate-fade" style={{ color: '#94a3b8', fontSize: 13, fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: 8 }}>
             <div className="spinner" style={{ width: 10, height: 10, borderColor: '#00DFD8', borderRightColor: 'transparent' }} />
             Optuna generating NSGA-II offspring...
           </div>
        )}
      </div>

      {activeState === 'done' && (
        <div style={{ marginTop: 10 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
             <div style={{ background: 'var(--bg-base)', padding: 12, borderRadius: 'var(--radius)' }}>
               <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Selected Engine</div>
               <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--primary)' }}>{bestConfig.model_name || 'N/A'}</div>
             </div>
             <div style={{ background: 'var(--bg-base)', padding: 12, borderRadius: 'var(--radius)' }}>
               <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Highest Signal</div>
               <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)' }}>
                 {new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(bestMetrics.primary_metric || 0)}
               </div>
             </div>
          </div>
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', background: 'var(--gradient-primary)', border: 'none' }}
            onClick={() => navigate(`/explainability?exp=${activeExpId}`)}
          >
            Launch Deep Interpretability Map
            <Icons.Eye />
          </button>
        </div>
      )}
    </div>
  );

  return (
    <div className="page-container animate-fade" style={{ maxWidth: 1400 }}>
      <header className="page-header" style={{ textAlign: 'center', marginBottom: 40 }}>
        <h1 className="page-title text-gradient-primary" style={{ fontSize: 40 }}>Live Execution Environment</h1>
        <p className="page-subtitle" style={{ fontSize: 18 }}>
          Watch the multi-agent swarm design, evaluate, and iteratively refine pipelines in real-time.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(600px, 1.5fr) 1fr', gap: 40, height: '600px', alignItems: 'stretch' }}>
        
        {/* Left Side: The Interactive Visual Flow Diagram */}
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <AgenticFlowDiagram activeState={activeState} />
          
          <div style={{ marginTop: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, display: 'flex', justifyContent: 'center', gap: 20 }}>
            <span><span style={{ color: 'var(--accent)', fontWeight: 800 }}>M</span> Meta-Feature Profiler</span>
            <span><span style={{ color: '#FF0080', fontWeight: 800 }}>A</span> Llama-3 Architect Agent</span>
            <span><span style={{ color: '#00DFD8', fontWeight: 800 }}>O</span> NSGA-II Optimizer</span>
            <span><span style={{ color: '#F9CB28', fontWeight: 800 }}>C</span> Llama-3 Critic Agent</span>
          </div>
        </div>

        {/* Right Side: Context / Action Pane */}
        <div style={{ height: '480px' }}>
          {activeExpId || uploadMutation.isPending ? renderLiveStatus() : renderUploadForm()}
        </div>

      </div>
    </div>
  );
}
