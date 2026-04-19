import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { api } from '../api/client';
import { Icons } from '../components/Icons';

export default function Upload() {
  const [file, setFile] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [description, setDescription] = useState('');
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: datasetsData } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.listDatasets,
  });
  const datasets = datasetsData?.datasets || [];

  const uploadMutation = useMutation({
    mutationFn: () => {
      // Note: Stage 1 added dataset_description as a form field
      return api.uploadDataset(file, targetCol, taskType, description);
    },
    onSuccess: (data) => {
      toast.success(`Success: ${data.name} initialized.`);
      qc.invalidateQueries(['datasets']);
      qc.invalidateQueries(['meta-db-stats']);
      navigate(`/dataset/${data.dataset_id}`);
    },
    onError: (err) => {
      toast.error(err.response?.data?.detail || 'System error during upload');
    },
  });

  const onDrop = useCallback((files) => {
    if (files[0]) {
      setFile(files[0]);
      // Smart default for target column if not set
      if (!targetCol) setTargetCol('target');
    }
  }, [targetCol]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  return (
    <div className="page-container animate-fade">
      <header className="page-header">
        <h1 className="page-title">Research Data Ingestion</h1>
        <p className="page-subtitle">
          Upload your experimental dataset to trigger hierarchical meta-profiling and semantic analysis.
        </p>
      </header>

      <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) 380px', alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
          {/* Upload Area */}
          <div
            {...getRootProps()}
            style={{
              padding: '80px 40px',
              border: '2px dashed var(--border)',
              borderRadius: 'var(--radius-xl)',
              background: isDragActive ? 'var(--accent-light)' : 'var(--bg-surface)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'var(--transition)',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            <input {...getInputProps()} />
            <div style={{ 
              color: file ? 'var(--success)' : 'var(--accent)', 
              marginBottom: 20,
              display: 'flex',
              justifyContent: 'center'
            }}>
              {file ? <Icons.CheckCircle size={56} /> : <Icons.Upload size={56} />}
            </div>
            
            {file ? (
              <div>
                <h3 style={{ fontSize: 20, fontWeight: 700, color: 'var(--primary)' }}>{file.name}</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>
                  {(file.size / 1024).toFixed(1)} KB — System Validated
                </p>
              </div>
            ) : (
              <div>
                <h3 style={{ fontSize: 20, fontWeight: 700, color: 'var(--primary)' }}>
                  {isDragActive ? 'Drop to Ingest' : 'Select Research Dataset'}
                </h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>
                  Drag & drop CSV files or click to browse local storage
                </p>
              </div>
            )}
          </div>

          {/* Configuration */}
          {file && (
            <div className="card animate-fade">
              <h3 className="card-title">
                <Icons.Settings size={20} />
                Ingestion Configuration
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
                <div className="form-group">
                  <label className="form-label">Target Variable</label>
                  <input
                    className="form-input"
                    placeholder="e.g. target_class"
                    value={targetCol}
                    onChange={(e) => setTargetCol(e.target.value)}
                  />
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    The column to be predicted by the synthesized pipeline.
                  </p>
                </div>

                <div className="form-group">
                  <label className="form-label">Analytical Task</label>
                  <select
                    className="form-select"
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: 32 }}>
                <label className="form-label">Semantic Description (Optional)</label>
                <textarea
                  className="form-input"
                  style={{ minHeight: '100px', resize: 'vertical' }}
                  placeholder="Describe the research context (e.g., 'Predicting early-stage churn for fintech subscribers using monthly transaction logs')"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Provided context enables LLM-driven search space synthesis and semantic similarity routing.
                </p>
              </div>

              <button
                className="btn btn-primary btn-lg"
                style={{ width: '100%' }}
                onClick={() => uploadMutation.mutate()}
                disabled={!targetCol || uploadMutation.isPending}
              >
                {uploadMutation.isPending ? 'Processing Matrix...' : 'Execute Ingestion Pipeline'}
                <Icons.ArrowRight />
              </button>
            </div>
          )}
        </div>

        {/* Sidebar / Stats */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div className="card">
            <h3 className="card-title">
              <Icons.Box size={18} />
              Recent Assets
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {datasets.slice(0, 5).map((ds) => (
                <div 
                  key={ds.id} 
                  style={{ 
                    padding: '12px', 
                    borderRadius: 'var(--radius)', 
                    border: '1px solid var(--border)',
                    cursor: 'pointer',
                    transition: 'var(--transition)'
                  }}
                  className="card-hover-bright"
                  onClick={() => navigate(`/dataset/${ds.id}`)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{ds.name}</span>
                    <Icons.ArrowRight size={14} color="var(--text-muted)" />
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    {ds.n_samples} samples &bull; {ds.task_type}
                  </div>
                </div>
              ))}
              {datasets.length === 0 && (
                <p style={{ fontSize: 13, color: 'var(--text-muted)', textAlign: 'center', padding: '20px 0' }}>
                  No datasets ingested yet.
                </p>
              )}
            </div>
          </div>

          <div className="card" style={{ background: 'var(--accent-light)', borderColor: 'transparent' }}>
            <h3 className="card-title" style={{ color: 'var(--accent)' }}>
              <Icons.Shield size={18} />
              Security & Privacy
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              All uploaded data is stored locally in the isolated research environment. 
              Meta-features are extracted in-situ, and semantic analysis uses local transformer models 
              to ensure air-gapped security for sensitive enterprise data.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
