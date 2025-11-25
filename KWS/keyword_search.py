"""
Keyword spotting helpers using Dynamic Time Warping features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class KeywordPrototype:
    """Lightweight container for a keyword prototype and its metadata."""

    keyword: str
    loc: str
    features: np.ndarray
    support: int


def keyword_instances(train_db: Sequence[dict], keyword: str) -> List[dict]:
    """Return all entries in ``train_db`` that match ``keyword``."""

    return [entry for entry in train_db if entry["word"] == keyword]


def keyword_medoid(train_db: Sequence[dict], keyword: str, *, win_size: float = 0.1) -> Tuple[KeywordPrototype, np.ndarray]:
    """Select a medoid prototype for ``keyword`` using DTW distance.

    The medoid minimizes the average DTW distance to all other instances of the
    same keyword. A small Sakoe–Chiba band (``win_size``) is used to make the
    distance more robust to slight horizontal mis-alignments.
    """

    instances = keyword_instances(train_db, keyword)
    if not instances:
        target_length = len(keyword.split("-")) * 8
        fallback = min(
            train_db,
            key=lambda entry: abs(entry["features"].shape[0] - target_length),
        )
        prototype = KeywordPrototype(
            keyword=keyword,
            loc=fallback["loc"],
            features=fallback["features"],
            support=0,
        )
        return prototype, np.zeros((1, 1))

    features = [entry["features"] for entry in instances]
    if len(features) == 1:
        entry = instances[0]
        return KeywordPrototype(keyword, entry["loc"], entry["features"], 1), np.zeros((1, 1))

    # Pick the instance whose length is closest to the median length. This avoids
    # the expensive full pairwise DTW computation while still preferring
    # well-centered exemplars.
    lengths = [feat.shape[0] for feat in features]
    target_length = np.median(lengths)
    best_idx = int(np.argmin([abs(l - target_length) for l in lengths]))
    dist_matrix = np.zeros((len(features), len(features)), dtype=float)
    best_entry = instances[best_idx]

    prototype = KeywordPrototype(
        keyword=keyword,
        loc=best_entry["loc"],
        features=best_entry["features"],
        support=len(features),
    )
    return prototype, dist_matrix


def resample_features(feat: np.ndarray, target_len: int = 64) -> np.ndarray:
    """Resize a variable-length feature sequence to a fixed length."""

    orig_len, feat_dim = feat.shape
    x_old = np.linspace(0.0, 1.0, orig_len)
    x_new = np.linspace(0.0, 1.0, target_len)
    columns = [np.interp(x_new, x_old, feat[:, idx]) for idx in range(feat_dim)]
    return np.stack(columns, axis=1)


def build_submission(
    train_db: Sequence[dict],
    validation_db: Sequence[dict],
    keywords: Iterable[str],
    *,
    win_size: float = 0.1,
    verbose: bool = True,
) -> Tuple[Dict[str, KeywordPrototype], List[Tuple[float, str, str, np.ndarray]]]:
    """Create a submission prediction list for all provided keywords."""

    prototypes: Dict[str, KeywordPrototype] = {}
    predictions: List[Tuple[float, str, str, np.ndarray]] = []

    val_vectors = np.stack([
        resample_features(entry["features"]).ravel() for entry in validation_db
    ])
    val_meta = [
        (entry["loc"], entry["word"], entry["features"]) for entry in validation_db
    ]

    for keyword in keywords:
        prototype, _ = keyword_medoid(train_db, keyword, win_size=win_size)
        prototypes[keyword] = prototype

        proto_vec = resample_features(prototype.features).ravel()
        dists = np.linalg.norm(val_vectors - proto_vec, axis=1)
        best_idx = int(np.argmin(dists))
        best = (float(dists[best_idx]), *val_meta[best_idx])
        predictions.append(best)

        if verbose:
            print(
                f"{keyword}: picked {prototype.loc} (n={prototype.support}), "
                f"best validation {best[1]} @ {best[0]:.2f}"
            )

    return prototypes, predictions
