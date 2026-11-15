"""
domains/cwru/rag_kb/seed_kb.py
===============================
Seeds the ChromaDB knowledge base for the CWRU Bearing Fault domain.

Run once (or re-run to refresh):
    python domains/cwru/rag_kb/seed_kb.py

The knowledge base contains domain facts about CWRU bearing fault signatures,
feature interpretation guidelines, and fault-specific decision rules.  These
are retrieved by the LLM agent (Layer 4) when it reasons about escalated events.

Knowledge categories
---------------------
1. Dataset overview       — CWRU dataset origin, classes, channels
2. Fault signatures       — vibration characteristics per fault class
3. Feature interpretation — what each of the 9 features means for bearings
4. Decision guidelines    — rules for classifying based on feature patterns
5. CWRU fault frequencies — characteristic defect frequencies (BPFI/BPFO/BSF)
6. Severity indicators    — how feature magnitudes map to fault severity
"""

from __future__ import annotations

import os
import sys

# Allow running from repo root: python domains/cwru/rag_kb/seed_kb.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chromadb

# ---------------------------------------------------------------------------
# Knowledge documents
# ---------------------------------------------------------------------------

_DOCUMENTS = [
    # ---- Dataset overview ----
    (
        "cwru_overview",
        "CWRU Bearing Fault Dataset Overview: "
        "The Case Western Reserve University (CWRU) bearing fault dataset is a "
        "widely-used benchmark for bearing fault detection. It contains vibration "
        "signals from bearings with four health conditions: (1) normal/healthy, "
        "(2) inner_race fault — defect on the inner raceway, (3) ball fault — "
        "defect on the rolling element, (4) outer_race fault — defect on the outer "
        "raceway. Signals are sampled at 12 kHz (drive-end) using an accelerometer "
        "mounted on the motor housing. Files include X<id>_DE_time (drive-end), "
        "X<id>_FE_time (fan-end), and X<id>_BA_time (base accelerometer, fault files only)."
    ),
    # ---- Normal bearing characteristics ----
    (
        "normal_signature",
        "Normal Bearing Vibration Signature: "
        "A healthy bearing exhibits low kurtosis (close to 3 for Gaussian noise, "
        "i.e. Fisher kurtosis near 0), low crest factor (typically < 3), and low "
        "impulse factor. The vibration signal is broadband noise with no periodic "
        "impulsive patterns. RMS and standard deviation are stable over time. "
        "Peak-to-peak amplitude is small relative to fault conditions. "
        "If kurtosis < 1 (Fisher), crest_factor < 3, and RMS is in its baseline "
        "range, classify as normal with high confidence."
    ),
    # ---- Inner race fault signature ----
    (
        "inner_race_signature",
        "Inner Race Fault Vibration Signature: "
        "Inner race faults generate periodic impacts at the Ball Pass Frequency "
        "Inner (BPFI). BPFI depends on shaft speed (RPM) and bearing geometry — "
        "for typical CWRU test bearings (6205-2RS): BPFI ≈ 5.415 × RPM/60 Hz. "
        "Key vibration features: (1) Elevated kurtosis > 3 (Fisher > 0) due to "
        "impulsive impacts, (2) High crest factor (> 4-6 at early stages, "
        "> 10 at advanced stages), (3) High impulse factor, (4) Modulated by "
        "shaft speed — sidebands around BPFI. The drive-end (DE) channel shows "
        "strong signatures. Inner race faults are the most common type and produce "
        "the strongest high-frequency energy. Classify as inner_race when kurtosis "
        "is elevated (Fisher > 1.5), crest_factor > 4, and impulse_factor > 5."
    ),
    # ---- Ball fault signature ----
    (
        "ball_signature",
        "Ball (Rolling Element) Fault Vibration Signature: "
        "Ball faults generate impacts at the Ball Spin Frequency (BSF). "
        "For CWRU 6205-2RS bearings: BSF ≈ 2.357 × RPM/60 Hz. "
        "Key vibration features: (1) Moderate kurtosis increase (more gradual "
        "than inner/outer race), (2) Crest factor elevated (3-7), (3) Modulated "
        "by cage frequency (FTF ≈ 0.3983 × RPM/60 Hz). Ball faults are often "
        "harder to detect because the contact point alternates between inner and "
        "outer races. The fan-end (FE) channel may also show elevated energy. "
        "Ball faults typically show skewness changes and elevated peak-to-peak "
        "values. Classify as ball when kurtosis Fisher > 1.0, crest_factor 3-7, "
        "and shape_factor indicates asymmetric distribution."
    ),
    # ---- Outer race fault signature ----
    (
        "outer_race_signature",
        "Outer Race Fault Vibration Signature: "
        "Outer race faults generate periodic impacts at the Ball Pass Frequency "
        "Outer (BPFO). For CWRU 6205-2RS bearings: BPFO ≈ 3.585 × RPM/60 Hz. "
        "Key vibration features: (1) High kurtosis due to stationary impulsive "
        "contacts (Fisher > 2 for moderate faults), (2) Very high crest factor "
        "(> 6 for developed faults), (3) Outer race faults are position-dependent "
        "— load zone affects severity. The outer race defect produces consistent "
        "periodic impacts because the defect is stationary. This often produces "
        "the clearest spectral signatures. Classify as outer_race when kurtosis "
        "Fisher > 2, crest_factor > 6, and peak/peak_to_peak values are elevated."
    ),
    # ---- Feature interpretation ----
    (
        "feature_rms_std",
        "Feature Interpretation — RMS and Standard Deviation: "
        "RMS (root-mean-square) measures the overall energy level of the vibration "
        "signal. For bearings, RMS increases with fault severity. It is insensitive "
        "to impulsive events early in fault development but rises significantly at "
        "advanced stages. Standard deviation measures the spread of vibration "
        "amplitude around the mean. Both RMS and std increase as fault severity "
        "grows. Sudden jumps in RMS or std relative to baseline suggest fault "
        "progression. Baseline normal RMS is typically 0.1-0.5 g for the CWRU "
        "test rig at 1750-1797 RPM."
    ),
    (
        "feature_kurtosis_crest",
        "Feature Interpretation — Kurtosis and Crest Factor: "
        "Kurtosis is the most sensitive early-fault indicator for bearings. "
        "It measures the 'tailedness' of the vibration amplitude distribution. "
        "Fisher kurtosis: 0 for Gaussian (normal bearing), positive values "
        "indicate impulsive peaks (faults). Values > 1.0 suggest fault onset, "
        "> 3.0 indicate moderate fault, > 6.0 indicate severe fault. "
        "Kurtosis can decrease at very severe faults when the signal becomes "
        "continuously rough (kurtosis 'reversal'). "
        "Crest factor (peak/RMS) complements kurtosis: healthy < 3, early fault "
        "3-6, moderate fault 6-10, severe fault > 10."
    ),
    (
        "feature_impulse_shape",
        "Feature Interpretation — Impulse Factor and Shape Factor: "
        "Impulse factor (peak / mean|x|) is very sensitive to impulsive events "
        "in early bearing fault stages. Normal bearings: impulse_factor 2-4. "
        "Faulty bearings: impulse_factor > 5, can reach 10-30 for severe faults. "
        "Shape factor (RMS / mean|x|) is a dimensionless measure of waveform "
        "shape. For sinusoidal waves it equals π/(2√2) ≈ 1.11, for random noise "
        "it is close to 1.25. Fault-induced impacts alter the shape factor away "
        "from baseline. Combined with kurtosis and impulse factor, it provides a "
        "reliable early-fault indicator."
    ),
    (
        "feature_peak_p2p_skewness",
        "Feature Interpretation — Peak, Peak-to-Peak, and Skewness: "
        "Peak value (max absolute amplitude) captures the largest single vibration "
        "event in the window. It is sensitive to individual impacts but can be "
        "noisy. Peak-to-peak (max - min) is useful for detecting asymmetric faults "
        "where the signal is biased in one direction. Skewness measures the "
        "asymmetry of the amplitude distribution. Normal bearings have near-zero "
        "skewness. Bearing faults — especially those producing one-sided impacts "
        "(common in inner race faults) — produce non-zero skewness. |skewness| > "
        "0.5 warrants attention in conjunction with elevated kurtosis."
    ),
    # ---- Decision guidelines ----
    (
        "decision_guidelines",
        "CWRU Bearing Fault Classification Guidelines: "
        "Use the following decision rules as starting points (not hard thresholds — "
        "adapt based on baseline for each test condition): "
        "1. Normal: Fisher kurtosis < 1.0, crest_factor < 3.5, impulse_factor < 4.0. "
        "2. Inner race (most impulsive): Fisher kurtosis > 1.5, crest_factor > 4, "
        "   impulse_factor > 5. DE channel dominates. "
        "3. Ball fault: Fisher kurtosis 1.0-2.5, crest_factor 3-7, shape_factor "
        "   deviation from baseline. FE channel contribution notable. "
        "4. Outer race (cleanest pattern): Fisher kurtosis > 2.0, crest_factor > 5, "
        "   peak_to_peak significantly elevated. Consistent periodic impacts. "
        "When features fall in ambiguous ranges, prefer to classify as the nearest "
        "fault class rather than 'normal'. False negatives (missed faults) are "
        "more costly than false alarms in industrial monitoring."
    ),
    # ---- Fault frequencies ----
    (
        "fault_frequencies",
        "CWRU Bearing Characteristic Defect Frequencies (6205-2RS JEM SKF): "
        "These frequencies scale linearly with shaft speed (RPM): "
        "BPFI (Ball Pass Freq Inner) = 5.4152 × (RPM/60) Hz — inner race fault. "
        "BPFO (Ball Pass Freq Outer) = 3.5848 × (RPM/60) Hz — outer race fault. "
        "BSF  (Ball Spin Freq)        = 2.3574 × (RPM/60) Hz — ball fault. "
        "FTF  (Fundamental Train Freq) = 0.3983 × (RPM/60) Hz — cage frequency. "
        "At 1750 RPM: BPFI≈158 Hz, BPFO≈104.5 Hz, BSF≈68.8 Hz, FTF≈11.6 Hz. "
        "At 1797 RPM: BPFI≈162 Hz, BPFO≈107.4 Hz, BSF≈70.7 Hz, FTF≈11.9 Hz. "
        "Statistical features in the time domain correlate with these frequencies "
        "because impacts at these rates produce characteristic kurtosis/crest factor spikes."
    ),
    # ---- Severity scale ----
    (
        "severity_scale",
        "CWRU Bearing Fault Severity Scale: "
        "The CWRU dataset includes faults at three severity levels (0.007', 0.014', "
        "0.021' defect diameter). Feature escalation across severities: "
        "Mild (0.007'): kurtosis Fisher 0.5-2.0, crest_factor 3-5, modest RMS increase. "
        "Moderate (0.014'): kurtosis Fisher 2-5, crest_factor 5-9, notable RMS increase. "
        "Severe (0.021'): kurtosis Fisher > 4 (may decrease if 'kurtosis reversal'), "
        "crest_factor > 8, significant RMS increase. "
        "Files 105-108 (inner_race), 130-133 (ball/outer_race) cover these severities. "
        "When the agent classifies a fault, it should also note approximate severity "
        "based on feature magnitudes."
    ),
    # ---- Multi-channel guidance ----
    (
        "multichannel_guidance",
        "CWRU Multi-Channel Vibration Analysis: "
        "Drive-End (DE) accelerometer is mounted closest to the test bearing and "
        "is the primary diagnostic channel. Fan-End (FE) accelerometer captures "
        "vibration transmitted through the shaft. BA (base) accelerometer is on "
        "the motor base plate and shows attenuated fault signatures. "
        "In the NeuroVerse CWRU pipeline, DE features (prefix 'de_') carry more "
        "diagnostic weight than FE features (prefix 'fe_'). "
        "Consistency between DE and FE channels increases classification confidence: "
        "if both channels show elevated kurtosis, the fault is real. "
        "Discrepancy between channels (high DE, low FE) may indicate measurement "
        "noise rather than a true fault."
    ),
]

# ---------------------------------------------------------------------------
# Seeder function
# ---------------------------------------------------------------------------


def seed_knowledge_base(kb_path: str | None = None) -> int:
    """
    Seed the CWRU ChromaDB knowledge base with bearing fault domain knowledge.

    Parameters
    ----------
    kb_path : str | None  Path to ChromaDB persist directory.
                          Defaults to the rag_kb/ folder next to this file.

    Returns
    -------
    int  Number of documents added (or already present).
    """
    if kb_path is None:
        kb_path = os.path.join(os.path.dirname(__file__))

    client = chromadb.PersistentClient(path=kb_path)
    collection = client.get_or_create_collection(
        name="cwru_knowledge",
        metadata={"description": "CWRU bearing fault domain knowledge"},
    )

    # Only add documents that are not already present
    existing_ids = set(collection.get()["ids"])
    new_docs = [(doc_id, text) for doc_id, text in _DOCUMENTS if doc_id not in existing_ids]

    if new_docs:
        collection.add(
            documents=[text for _, text in new_docs],
            ids=[doc_id for doc_id, _ in new_docs],
            metadatas=[{"source": "cwru_domain_knowledge"} for _ in new_docs],
        )
        print(f"[seed_kb] Added {len(new_docs)} documents to {kb_path!r}.")
    else:
        print(f"[seed_kb] Knowledge base already seeded ({len(existing_ids)} docs).")

    total = collection.count()
    print(f"[seed_kb] Total documents in collection: {total}")
    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed the CWRU ChromaDB knowledge base."
    )
    parser.add_argument(
        "--kb-path",
        default=None,
        help="Path to ChromaDB directory (default: domains/cwru/rag_kb/)",
    )
    args = parser.parse_args()
    seed_knowledge_base(args.kb_path)
