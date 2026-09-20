# MetaLearnX

**LLM-assisted AutoML with a Python API and React dashboard**

MetaLearnX explores how language-model recommendations can guide tabular machine-learning pipelines. It combines dataset profiling, pipeline construction, Optuna search, explanations, and experiment tracking.

This is a research and development project. Performance, reliability, and deployment readiness must be evaluated for the intended dataset and environment.

## How it works

1. **Profile the dataset.** Extract statistical features and identify missingness, imbalance, and other characteristics.
2. **Propose a pipeline.** An architect agent requests a structured configuration through LiteLLM. The orchestrator has a fallback when the architect cannot produce a proposal.
3. **Evaluate configurations.** Optuna searches model parameters. Multi-objective mode uses NSGA-II over predictive score, training time, model size, and inference latency. Single-objective mode supports TPE, random search, and CMA-ES.
4. **Review and explain.** The orchestrator combines agent feedback with evaluation and explanation modules.
5. **Record experiments.** SQLite stores dataset profiles, configurations, metrics, and reasoning logs.

LLM recommendations are heuristics. Their value needs to be established through held-out evaluation and comparisons with suitable baselines.

## Source map

| Component | Entry point |
| --- | --- |
| API | [apps/api/main.py](apps/api/main.py) |
| Workflow | [core/orchestrator.py](core/orchestrator.py) |
| Dataset profiling | [core/dataset_intelligence/profiler.py](core/dataset_intelligence/profiler.py) |
| Structured agent calls | [core/agents/agent_base.py](core/agents/agent_base.py) |
| Search and selection | [core/optimization/optimizer.py](core/optimization/optimizer.py) |
| Experiment storage | [core/tracking/meta_db.py](core/tracking/meta_db.py) |
| Dashboard | [apps/dashboard](apps/dashboard) |
| Tests | [tests/test_core.py](tests/test_core.py) |
| Experimental extensions | [core/frontier](core/frontier) |

## Local setup

### Requirements

- Python 3.10+; dependency compatibility depends on the versions installed from [requirements.txt](requirements.txt).
- Node.js 20.19+ within the 20.x line, or 22.12+; these are the engine constraints in the locked Vite dependency.
- Provider credentials for live LLM calls. The checked-in agent defaults reference `groq/llama3-70b-8192`; verify provider availability and configure an available model before using those calls.

The commands below match the repository layout and configuration. They are not a report of a successful installation on every supported platform.

### Backend

```bash
git clone https://github.com/bnssaanirudh/MetaLearnX.git
cd MetaLearnX
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

Then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

For Groq-backed calls, set `GROQ_API_KEY` in the backend process environment before starting it. Keep credentials out of committed files. Agent model defaults are defined in [core/agents](core/agents).

### Dashboard

In a second terminal, starting from the repository root:

```bash
cd apps/dashboard
npm ci
npm run dev
```

Open the address printed by Vite, normally `http://localhost:3000`. The development configuration proxies `/api` to `http://localhost:8000`.

## Verification

From the repository root, with Python dependencies installed:

```bash
python -m pytest tests/test_core.py
```

From `apps/dashboard`:

```bash
npm run lint
npm run build
```

Passing unit tests alone does not establish end-to-end agent reliability or model quality. Evaluate the full workflow with representative data and record the environment, split, seed, search budget, and provider configuration.

## Technical boundaries

- **Structured output:** the base agent makes at most three attempts and validates responses with Pydantic. On validation failure it adds feedback and halves the temperature. This does not guarantee valid output, semantic correctness, or determinism; exhaustion returns `None`.
- **Search methods:** NSGA-II is evolutionary multi-objective optimization. The single-objective TPE option is a separate search strategy. Neither validates an LLM recommendation without empirical evaluation.
- **Model selection:** the balanced Pareto selector uses normalized distances and an opacity penalty. It does not implement a universal rule guaranteeing a simpler model whenever its score reaches 98% of another model's score.
- **Storage:** SQLite WAL supports concurrent access patterns. The experiment database is mutable and is not a tamper-proof or immutable audit log.
- **Metrics:** the function named `_compute_hypervolume` currently computes a normalized volume proxy, not an exact dominated hypervolume indicator. Interpret it accordingly.
- **Experimental scope:** modules in `core/frontier` are experimental and may depend on optional packages. Their presence is not evidence of validated quantum advantage, cryptographic security, causal identification, or deployment readiness.
- **Benchmark claims:** no fixed search-space reduction or superiority over other AutoML systems is claimed here. Such statements require linked experiments with comparable data splits and compute budgets.
