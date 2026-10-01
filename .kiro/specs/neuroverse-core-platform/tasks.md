# Tasks: NeuroVerse Core Platform

## Task list

- [ ] 1. Scaffold repo layout
  - Create `/platform/__init__.py`, `/domains/traffic/__init__.py`, `/tests/` directory
  - Add `requirements.txt` with pinned versions for new deps (langgraph, langchain-anthropic, chromadb, jsonschema, pytest)
  - _Requirement: Req 6.3 (no traffic imports in /platform), Req 5.3_

- [ ] 2. Implement `domain_plugin.py` — abstract base class
  - Define `DomainPlugin` ABC with four abstract properties and `validate_event()`
  - Write `tests/test_domain_plugin.py`: verify direct instantiation raises `TypeError`
  - _Requirement: Req 5.1–5.4_

- [ ] 3. Implement `snn_plugin.py` — SNN input plugin
  - Implement `SNNInputPlugin` with snnTorch LIFCell default backend
  - Add stub classes for SpikingJelly and Norse (raise `NotImplementedError`)
  - Write `tests/test_snn_plugin.py`: backend swap, shape check
  - _Requirement: Req 1.1–1.5_

- [ ] 4. Implement `implicit_layer.py` — predictive layer + ReasoningScore
  - Implement `ReasoningScore` dataclass
  - Implement `ImplicitPredictiveLayer` with `_PrototypeStore`, `_MCDropoutHead`, `_PredictorHead`, `_EpisodicMemory`
  - Implement `compute_state()` and `fit()`
  - Write `tests/test_implicit_layer.py`: fixed input → all fields in [0, 1]
  - _Requirement: Req 2.1–2.6_

- [ ] 5. Implement `gate.py` — reasoning gate
  - Implement `EscalationHistory` dataclass and `ReasoningGate` class
  - Wire `GateLearner.predict_proba()` as the decision function
  - Implement decision logging and `log_outcome()`
  - Write `tests/test_gate.py`: low-score batch, all-1.0 batch, log growth
  - _Requirement: Req 3.1–3.8_

- [ ] 6. Implement `gate_learner.py` — meta-learner
  - Implement `GateLearner` with logistic regression over 7 features (4 sub-signals + 3 history)
  - Weighted cross-entropy loss: FN weight = 5.0, FP weight = 1.0
  - Auto-retrain trigger at `len(log) % UPDATE_EVERY == 0`
  - _Requirement: Req 3.2–3.6_

- [ ] 7. Implement `agent_plugin.py` — LLM agent plugin
  - Implement `AgentDecision` dataclass
  - Build LangGraph ReAct graph with `rag_retrieve` and `lookup_context` tools
  - Implement timeout + graceful degradation fallback
  - Write `tests/test_agent_plugin.py`: mock LLM + ChromaDB, fallback on timeout
  - _Requirement: Req 4.1–4.7_

- [ ] 8. Implement traffic domain plugin
  - Create `/domains/traffic/domain.py` — `TrafficDomain(DomainPlugin)` with correct schema
  - Define `input_schema` matching `preprocess.py` feature columns (12 lane features + crash_record)
  - Point `knowledge_base_path` to `domains/traffic/rag_kb/`
  - _Requirement: Req 5.3, Req 6.3_

- [ ] 9. Write end-to-end smoke test
  - `tests/test_e2e_smoke.py`: load one batch from `data/processed/train.pt`, run full pipeline
  - Assert no crash, print summary line
  - `ast.walk` import check: no domain imports in `/platform/` files
  - _Requirement: Req 6.1–6.3_

- [ ] 10. Run full test suite and verify gate metric
  - `pytest tests/ -v`
  - Confirm gate reduces escalations ≥30% vs. always-escalate on val set (report in test output)
  - Fix any failures
  - _Requirement: Req 3.5, all unit test requirements_
