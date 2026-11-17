"""
Seed the traffic domain ChromaDB knowledge base with crash detection rules.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def seed_knowledge_base(kb_path: str) -> None:
    """Populate the ChromaDB knowledge base with traffic crash detection facts."""
    import chromadb

    client = chromadb.PersistentClient(path=kb_path)
    col = client.get_or_create_collection("knowledge")

    if col.count() > 0:
        return

    documents = [
        "A traffic crash is indicated by a sudden drop in lane speeds across "
        "multiple milemarkers simultaneously, combined with high speed variance "
        "between milemarkers (high spatial_speed_std).",

        "High spatial_occ_max with low spatial_speed_min suggests a congestion "
        "wave or crash front moving through the freeway network.",

        "Normal traffic flow shows low cross-milemarker speed variance and "
        "moderate occupancy. Lane speeds should be relatively uniform across "
        "milemarkers at the same timestamp.",

        "A high prediction_error signal from the SNN means the spike pattern "
        "was difficult to predict from the previous timestep, suggesting an "
        "abrupt change in traffic state.",

        "High novelty indicates the current traffic pattern does not match any "
        "stored prototype, which for a stable freeway is unusual and may "
        "indicate an incident.",

        "Risk is a composite signal combining novelty, uncertainty, "
        "prediction_error, and spatial_correlation. Risk > 0.6 strongly "
        "suggests an anomalous event worthy of escalation.",

        "The FT-AED dataset contains approximately 3.7 million radar samples "
        "with about 400 labeled crash events. Crashes are extremely rare "
        "(< 0.01% of windows), so the gate must be sensitive to avoid false "
        "negatives.",

        "Spatial correlation measures how anomalous the cross-sensor pattern "
        "is relative to the running baseline. A crash typically disrupts "
        "multiple lanes and milemarkers simultaneously, producing high spatial "
        "correlation.",
    ]

    ids = [f"traffic_doc_{i}" for i in range(len(documents))]
    col.add(documents=documents, ids=ids)
    print(f"Seeded traffic KB with {len(documents)} documents at {kb_path}")


if __name__ == "__main__":
    kb = str(Path(__file__).parent)
    seed_knowledge_base(kb)
