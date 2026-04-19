import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const http = axios.create({ baseURL: BASE, timeout: 120_000 });

export const api = {
  // System
  getHealth:      () => http.get('/health').then(r => r.data),
  getMetaDbStats: () => http.get('/api/meta-db/stats').then(r => r.data),

  // Datasets
  uploadDataset: (file, targetCol, taskType) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('target_col', targetCol);
    fd.append('task_type', taskType);
    return http.post('/api/datasets/upload', fd).then(r => r.data);
  },
  listDatasets:     () => http.get('/api/datasets').then(r => r.data),
  getDataset:       (id) => http.get(`/api/datasets/${id}`).then(r => r.data),
  getSimilar:       (id, k = 5) => http.get(`/api/datasets/${id}/similar`, { params: { top_k: k } }).then(r => r.data),

  // Recommendations
  getRecommendations: (id, taskType) =>
    http.get(`/api/recommend/${id}`, { params: { task_type: taskType } }).then(r => r.data),

  // Experiments
  runExperiment:      (payload) => http.post('/api/experiments/run', payload).then(r => r.data),
  getExpStatus:       (id) => http.get(`/api/experiments/${id}/status`).then(r => r.data),
  getExpResults:      (id) => http.get(`/api/experiments/${id}/results`).then(r => r.data),
  getExpExplain:      (id) => http.get(`/api/experiments/${id}/explain`).then(r => r.data),
  listExperiments:    (dsId) => http.get('/api/experiments/history', { params: dsId ? { dataset_id: dsId } : {} }).then(r => r.data),

  // Benchmark
  runBenchmark:       () => http.post('/api/benchmark/run').then(r => r.data),
  runAblation:        () => http.post('/api/benchmark/ablation').then(r => r.data),
  getBenchmarkResults:() => http.get('/api/benchmark/results').then(r => r.data),

  // Advanced: Uncertainty
  getUncertainRecommend: (id, nPasses = 50, taskType = 'classification') =>
    http.get(`/api/recommend/${id}/uncertain`, {
      params: { n_passes: nPasses, task_type: taskType }
    }).then(r => r.data),

  // Advanced: Few-Shot
  getFewShotAnalysis: (id) =>
    http.get(`/api/datasets/${id}/few-shot`).then(r => r.data),

  // Advanced: Knowledge Graph
  getKnowledgeGraph:     () => http.get('/api/knowledge-graph').then(r => r.data),
  rebuildKnowledgeGraph: () => http.post('/api/knowledge-graph/rebuild').then(r => r.data),

  // Advanced: Continual Learning
  getContinualStats: () => http.get('/api/continual-learning/stats').then(r => r.data),

  // Advanced: Search Space
  getSearchSpaceStats: () => http.get('/api/search-space/stats').then(r => r.data),

  // Advanced: Federated
  runFederated: (nRounds = 5) =>
    http.post('/api/federated/run', null, { params: { n_rounds: nRounds } }).then(r => r.data),
};
