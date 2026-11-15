"""
tests/test_cwru_domain.py
==========================
Unit and integration tests for the CWRU Bearing Fault Detection domain.

Covers:
  - CWRULoader  (loader.py)
  - CWRUPreprocessor (preprocess.py)
  - CWRUDomain (domain.py)
  - CWRUPipeline smoke test (pipeline.py)

Tests run without real .mat files by using synthetic numpy signals.
Integration tests that require the actual data files are marked with
``@pytest.mark.integration`` and are skipped in CI when data is absent.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.io
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mat_bytes(file_id: str, n: int = 2048) -> bytes:
    """Create a minimal valid CWRU .mat file in memory."""
    rng = np.random.default_rng(42)
    mdict = {
        f"X{file_id}_DE_time": rng.normal(0, 0.5, (n, 1)).astype(np.float64),
        f"X{file_id}_FE_time": rng.normal(0, 0.3, (n, 1)).astype(np.float64),
        f"X{file_id}RPM":      np.array([[1750]], dtype=np.uint16),
    }
    buf = io.BytesIO()
    scipy.io.savemat(buf, mdict)
    return buf.getvalue()


def _make_mat_zip_bytes(file_id: str, n: int = 2048) -> bytes:
    """Create a minimal .mat.zip in memory."""
    mat_bytes = _make_mat_bytes(file_id, n)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{file_id}.mat", mat_bytes)
    return buf.getvalue()


# ===========================================================================
# Loader tests
# ===========================================================================

class TestCWRULoader:

    def test_load_file_from_mat(self, tmp_path):
        from domains.cwru.loader import CWRULoader, FAULT_CLASSES

        mat_bytes = _make_mat_bytes("100", n=2048)
        mat_file = tmp_path / "100.mat"
        mat_file.write_bytes(mat_bytes)

        record = CWRULoader.load_file(mat_file, fault_class="normal")
        assert record.fault_class == "normal"
        assert record.label == 0
        assert record.file_id == "100"
        assert record.de_time.shape == (2048,)
        assert record.fe_time.shape == (2048,)
        assert record.rpm == 1750.0

    def test_load_file_from_zip(self, tmp_path):
        from domains.cwru.loader import CWRULoader

        zip_bytes = _make_mat_zip_bytes("105", n=2048)
        zip_file = tmp_path / "105.mat.zip"
        zip_file.write_bytes(zip_bytes)

        record = CWRULoader.load_file(zip_file, fault_class="inner_race")
        assert record.fault_class == "inner_race"
        assert record.label == 1
        assert record.de_time.shape == (2048,)

    def test_load_file_unknown_class_raises(self, tmp_path):
        from domains.cwru.loader import CWRULoader

        mat_bytes = _make_mat_bytes("99", n=512)
        mat_file = tmp_path / "99.mat"
        mat_file.write_bytes(mat_bytes)

        with pytest.raises(ValueError, match="Unknown fault_class"):
            CWRULoader.load_file(mat_file, fault_class="unknown_class")

    def test_load_file_missing_raises(self):
        from domains.cwru.loader import CWRULoader

        with pytest.raises(FileNotFoundError):
            CWRULoader.load_file("/nonexistent/file.mat", fault_class="normal")

    def test_load_all_discovers_files(self, tmp_path):
        from domains.cwru.loader import CWRULoader

        # Create class subdirectories with synthetic files
        for cls, fid in [("normal", "100"), ("inner_race", "105"),
                          ("ball", "130"), ("outer_race", "131")]:
            d = tmp_path / cls
            d.mkdir()
            if cls == "normal":
                (d / f"{fid}.mat").write_bytes(_make_mat_bytes(fid))
            else:
                (d / f"{fid}.mat.zip").write_bytes(_make_mat_zip_bytes(fid))

        loader = CWRULoader(data_root=tmp_path)
        records = loader.load_all()
        assert len(records) == 4

        classes_found = {r.fault_class for r in records}
        assert classes_found == {"normal", "inner_race", "ball", "outer_race"}

    def test_load_all_skips_corrupted(self, tmp_path):
        from domains.cwru.loader import CWRULoader
        import warnings

        d = tmp_path / "normal"
        d.mkdir()
        (d / "bad.mat").write_bytes(b"\xff\xfe" * 100)  # gibberish

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader = CWRULoader(data_root=tmp_path)
            records = loader.load_all()

        assert len(records) == 0
        assert any("skipping" in str(warning.message) for warning in w)

    def test_label_map(self):
        from domains.cwru.loader import LABEL_MAP, FAULT_CLASSES

        assert set(LABEL_MAP.keys()) == set(FAULT_CLASSES)
        assert LABEL_MAP["normal"] == 0
        assert LABEL_MAP["inner_race"] == 1
        assert LABEL_MAP["ball"] == 2
        assert LABEL_MAP["outer_race"] == 3


# ===========================================================================
# Preprocessor tests
# ===========================================================================

class TestCWRUPreprocessor:

    def _make_record(self, fault_class="normal", label=0, n=4096, has_fe=True):
        from domains.cwru.loader import CWRURecord

        rng = np.random.default_rng(0)
        return CWRURecord(
            fault_class=fault_class,
            label=label,
            file_id="test",
            source_path="/tmp/test.mat",
            de_time=rng.normal(0, 0.5, n),
            fe_time=rng.normal(0, 0.3, n) if has_fe else np.array([]),
            rpm=1750.0,
            sample_rate=12_000,
        )

    def test_window_count(self):
        from domains.cwru.preprocess import CWRUPreprocessor

        prep = CWRUPreprocessor(window_size=1024, overlap=0.5)
        record = self._make_record(n=4096)
        windows = prep.process_records([record])
        # With overlap=0.5, step=512, windows = (4096-1024)//512 + 1 = 7
        assert len(windows) == 7

    def test_feature_shape_de_fe(self):
        from domains.cwru.preprocess import CWRUPreprocessor, NUM_FEATURES_DE_FE

        prep = CWRUPreprocessor(window_size=1024, overlap=0.0, use_fe=True)
        record = self._make_record(has_fe=True)
        windows = prep.process_records([record])
        assert windows[0].features.shape == (NUM_FEATURES_DE_FE,)
        assert len(windows[0].feature_cols) == NUM_FEATURES_DE_FE

    def test_feature_shape_de_only(self):
        from domains.cwru.preprocess import CWRUPreprocessor, NUM_FEATURES_DE_ONLY

        prep = CWRUPreprocessor(window_size=1024, overlap=0.0, use_fe=True)
        record = self._make_record(has_fe=False)  # no FE
        windows = prep.process_records([record])
        assert windows[0].features.shape == (NUM_FEATURES_DE_ONLY,)

    def test_feature_values_finite(self):
        from domains.cwru.preprocess import CWRUPreprocessor

        prep = CWRUPreprocessor(window_size=512, overlap=0.0)
        record = self._make_record(n=2048)
        windows = prep.process_records([record])
        for w in windows:
            assert np.all(np.isfinite(w.features)), "Features contain non-finite values"

    def test_fault_has_higher_kurtosis_than_normal(self):
        """Faulty bearings should exhibit higher kurtosis in their DE features."""
        from domains.cwru.preprocess import CWRUPreprocessor
        from domains.cwru.loader import CWRURecord

        rng = np.random.default_rng(1)
        n = 8192
        window_size = 1024

        # Normal: Gaussian noise → low kurtosis
        normal_de = rng.normal(0, 0.5, n)
        # Faulty: Gaussian + periodic impulses → high kurtosis
        faulty_de = rng.normal(0, 0.3, n)
        # Add 15 random impulse spikes
        impulse_locs = rng.integers(0, n, 15)
        faulty_de[impulse_locs] += rng.choice([-1, 1], 15) * 5.0

        def _make(de, fc, lbl):
            return CWRURecord("n" if fc == "normal" else fc, lbl,
                               "t", "/t.mat", de, np.array([]), 1750.0, 12000)

        prep = CWRUPreprocessor(window_size=window_size, overlap=0.0, use_fe=False)
        normal_wins  = prep.process_records([_make(normal_de, "normal", 0)])
        faulty_wins  = prep.process_records([_make(faulty_de, "inner_race", 1)])

        # de_kurtosis is index 4 in _CHANNEL_FEATURES
        from domains.cwru.preprocess import _CHANNEL_FEATURES
        kurt_idx = _CHANNEL_FEATURES.index("kurtosis")

        normal_kurt = np.mean([w.features[kurt_idx] for w in normal_wins])
        faulty_kurt = np.mean([w.features[kurt_idx] for w in faulty_wins])
        assert faulty_kurt > normal_kurt, (
            f"Expected faulty kurtosis > normal kurtosis, "
            f"got faulty={faulty_kurt:.3f}, normal={normal_kurt:.3f}"
        )

    def test_to_events(self):
        from domains.cwru.preprocess import CWRUPreprocessor

        prep = CWRUPreprocessor(window_size=512, overlap=0.0)
        record = self._make_record(n=2048)
        windows = prep.process_records([record])
        events = prep.to_events(windows)
        assert len(events) == len(windows)
        assert "timestamp" in events[0]
        assert "fault_class" in events[0]
        assert "de_rms" in events[0]

    def test_normalize_events(self):
        from domains.cwru.preprocess import CWRUPreprocessor

        prep = CWRUPreprocessor(window_size=512, overlap=0.0)
        record = self._make_record(n=4096)
        windows = prep.process_records([record])
        events = prep.to_events(windows)
        normed = prep.normalize_events(events, fit=True)

        # After normalisation, feature values should have ~0 mean and ~1 std
        de_rms_vals = [e["de_rms"] for e in normed]
        assert abs(np.mean(de_rms_vals)) < 0.5  # roughly centered
        assert len(normed) == len(events)

    def test_overlap_raises_on_invalid(self):
        from domains.cwru.preprocess import CWRUPreprocessor

        with pytest.raises(ValueError):
            CWRUPreprocessor(window_size=1024, overlap=1.0)

        with pytest.raises(ValueError):
            CWRUPreprocessor(window_size=1024, overlap=-0.1)


# ===========================================================================
# Domain tests
# ===========================================================================

class TestCWRUDomain:

    @pytest.fixture
    def domain(self):
        from domains.cwru.domain import CWRUDomain
        return CWRUDomain()

    @pytest.fixture
    def de_only_domain(self):
        from domains.cwru.domain import CWRUDomain
        return CWRUDomain(de_only=True)

    def _valid_event(self, domain):
        from domains.cwru.domain import FEATURE_COLS
        ev = {"timestamp": 0.0}
        for col in domain.feature_cols:
            ev[col] = 0.0
        return ev

    def test_instantiates(self, domain):
        assert domain is not None

    def test_domain_name(self, domain):
        assert domain.domain_name() == "CWRUDomain"

    def test_input_schema_has_all_feature_cols(self, domain):
        schema = domain.input_schema
        for col in domain.feature_cols:
            assert col in schema["properties"], f"{col} missing from input_schema"
        assert "timestamp" in schema["properties"]

    def test_output_schema_has_required_fields(self, domain):
        schema = domain.output_schema
        assert "label" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_output_schema_labels(self, domain):
        labels = domain.output_schema["properties"]["label"]["enum"]
        assert "normal" in labels
        assert "inner_race" in labels
        assert "ball" in labels
        assert "outer_race" in labels
        assert "uncertain" in labels

    def test_threshold_hints_all_keys(self, domain):
        hints = domain.threshold_hints
        for key in ("novelty", "uncertainty", "prediction_error", "memory_confidence"):
            assert key in hints
            assert 0.0 <= hints[key] <= 1.0

    def test_validate_event_valid(self, domain):
        assert domain.validate_event(self._valid_event(domain)) is True

    def test_validate_event_missing_feature(self, domain):
        ev = self._valid_event(domain)
        del ev[domain.feature_cols[0]]
        assert domain.validate_event(ev) is False

    def test_validate_event_missing_timestamp(self, domain):
        ev = self._valid_event(domain)
        del ev["timestamp"]
        assert domain.validate_event(ev) is False

    def test_num_features_de_fe(self, domain):
        assert domain.num_features == 18

    def test_num_features_de_only(self, de_only_domain):
        assert de_only_domain.num_features == 9

    def test_event_to_tensor_shape(self, domain):
        ev = self._valid_event(domain)
        t = domain.event_to_tensor(ev)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 1, 18)
        assert t.dtype == torch.float32

    def test_event_to_tensor_de_only_shape(self, de_only_domain):
        ev = self._valid_event(de_only_domain)
        t = de_only_domain.event_to_tensor(ev)
        assert t.shape == (1, 1, 9)

    def test_label_to_class(self):
        from domains.cwru.domain import CWRUDomain
        assert CWRUDomain.label_to_class(0) == "normal"
        assert CWRUDomain.label_to_class(1) == "inner_race"
        assert CWRUDomain.label_to_class(2) == "ball"
        assert CWRUDomain.label_to_class(3) == "outer_race"
        assert CWRUDomain.label_to_class(99) == "unknown"

    def test_class_to_label(self):
        from domains.cwru.domain import CWRUDomain
        assert CWRUDomain.class_to_label("normal") == 0
        assert CWRUDomain.class_to_label("inner_race") == 1
        assert CWRUDomain.class_to_label("ball") == 2
        assert CWRUDomain.class_to_label("outer_race") == 3
        assert CWRUDomain.class_to_label("unknown") == -1

    def test_knowledge_base_path_is_string(self, domain):
        path = domain.knowledge_base_path
        assert isinstance(path, str)
        assert "rag_kb" in path

    def test_event_from_window(self, domain):
        window_dict = {
            "timestamp": 1.5,
            "fault_class": "inner_race",
            "label": 1,
            "file_id": "105",
            "window_idx": 3,
            "rpm": 1750.0,
            **{col: 0.5 for col in domain.feature_cols},
        }
        event = domain.event_from_window(window_dict)
        assert event["timestamp"] == 1.5
        assert event["fault_class"] == "inner_race"
        for col in domain.feature_cols:
            assert col in event


# ===========================================================================
# Pipeline smoke test
# ===========================================================================

class TestCWRUPipelineSmoke:
    """
    Smoke tests for the pipeline using synthetic in-memory data.
    Does not require actual CWRU .mat files.
    """

    def _make_tmp_data_root(self, tmp_path: Path) -> Path:
        """Create a minimal data_root with one file per class."""
        for cls, fid in [("normal", "100"), ("inner_race", "105"),
                          ("ball", "130"), ("outer_race", "131")]:
            d = tmp_path / cls
            d.mkdir()
            if cls == "normal":
                (d / f"{fid}.mat").write_bytes(_make_mat_bytes(fid, n=4096))
            else:
                (d / f"{fid}.mat.zip").write_bytes(_make_mat_zip_bytes(fid, n=4096))
        return tmp_path

    def test_pipeline_runs_no_llm(self, tmp_path):
        from domains.cwru.pipeline import CWRUPipeline

        data_root = self._make_tmp_data_root(tmp_path)

        pipe = CWRUPipeline(
            data_root=str(data_root),
            window_size=512,
            overlap=0.0,
            num_steps=5,
            enable_llm=False,
            max_events=20,
            seed_kb=False,
        )
        results = pipe.run()
        assert len(results) > 0
        assert len(results) <= 20

    def test_pipeline_results_have_expected_fields(self, tmp_path):
        from domains.cwru.pipeline import CWRUPipeline

        data_root = self._make_tmp_data_root(tmp_path)
        pipe = CWRUPipeline(
            data_root=str(data_root),
            window_size=512,
            overlap=0.0,
            num_steps=5,
            enable_llm=False,
            max_events=10,
            seed_kb=False,
        )
        results = pipe.run()
        for r in results:
            d = r.to_dict()
            assert "event_id" in d
            assert "fault_class" in d
            assert "escalated" in d
            assert "decision_label" in d
            assert "decision_conf" in d
            assert 0.0 <= d["decision_conf"] <= 1.0

    def test_pipeline_summary(self, tmp_path):
        from domains.cwru.pipeline import CWRUPipeline

        data_root = self._make_tmp_data_root(tmp_path)
        pipe = CWRUPipeline(
            data_root=str(data_root),
            window_size=512,
            overlap=0.0,
            num_steps=5,
            enable_llm=False,
            max_events=20,
            seed_kb=False,
        )
        results = pipe.run()
        summary = pipe.summarize(results)

        assert summary.total_events == len(results)
        assert 0.0 <= summary.escalation_rate <= 1.0
        assert summary.avg_latency_ms >= 0.0
        summary_dict = summary.to_dict()
        assert "total_events" in summary_dict
        assert "escalation_rate" in summary_dict

    def test_pipeline_raises_on_empty_data_root(self, tmp_path):
        from domains.cwru.pipeline import CWRUPipeline

        # Empty data root — no files
        empty = tmp_path / "empty"
        empty.mkdir()
        pipe = CWRUPipeline(
            data_root=str(empty),
            enable_llm=False,
            seed_kb=False,
        )
        with pytest.raises(RuntimeError, match="no records loaded"):
            pipe.run()

    def test_run_single_event(self, tmp_path):
        from domains.cwru.pipeline import CWRUPipeline
        from domains.cwru.domain import CWRUDomain

        data_root = self._make_tmp_data_root(tmp_path)
        pipe = CWRUPipeline(
            data_root=str(data_root),
            window_size=512,
            overlap=0.0,
            num_steps=5,
            enable_llm=False,
            max_events=5,
            seed_kb=False,
        )
        # Warm up implicit layer by running full pipeline first
        pipe.run()

        domain = CWRUDomain()
        event = {"timestamp": 0.0, **{col: 0.0 for col in domain.feature_cols}}
        result = pipe.run_single(event)
        assert result.decision_label in (
            "normal", "inner_race", "ball", "outer_race", "uncertain"
        )


# ===========================================================================
# Architecture invariant: CWRU domain must not import from /nv_platform
# inside its own domain files in a way that breaks the DomainPlugin contract
# ===========================================================================

class TestArchitectureInvariant:

    def test_cwru_imports_domain_plugin_base(self):
        """CWRUDomain must subclass DomainPlugin from nv_platform."""
        from domains.cwru.domain import CWRUDomain
        from nv_platform.domain_plugin import DomainPlugin

        assert issubclass(CWRUDomain, DomainPlugin)

    def test_cwru_satisfies_domain_plugin_contract(self):
        """All four abstract properties must be implemented."""
        from domains.cwru.domain import CWRUDomain

        domain = CWRUDomain()
        # These would raise TypeError at instantiation if abstract — but let's
        # also verify they return the right types
        assert isinstance(domain.input_schema, dict)
        assert isinstance(domain.output_schema, dict)
        assert isinstance(domain.knowledge_base_path, str)
        assert isinstance(domain.threshold_hints, dict)

    def test_platform_does_not_import_cwru(self):
        """
        Verify that no /nv_platform/ module imports from /domains/cwru/.
        This preserves the domain-agnostic architecture invariant.
        """
        platform_dir = Path(__file__).parents[1] / "nv_platform"
        for py_file in platform_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            assert "domains.cwru" not in source, (
                f"{py_file.name} imports from domains.cwru — "
                "this breaks the platform/domain separation."
            )
            assert "from domains" not in source, (
                f"{py_file.name} has a 'from domains' import — "
                "platform files must never import from domains."
            )
