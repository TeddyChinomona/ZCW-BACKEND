"""
zimcrimewatch/ml_utils.py
=========================
Machine-learning helpers for the ZimCrimeWatch backend.

Contents
--------
1. compute_kde_heatmap      — Kernel Density Estimation for the heatmap API
2. compute_time_series      — Seasonal decomposition for temporal analysis
3. ProfileMatcher           — Random Forest crime profile matcher
       .train()             — offline training (management command)
       .find_similar()      — FIX: was missing; returns IDs of the most
                              similar incidents to a given CrimeIncident
       .predict()           — group-label ranking for a free-text query
       .load() / _save()    — pickle persistence
4. compute_hotspot_summary  — DBSCAN spatial clustering for hotspot API
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Model storage path — lives next to this file inside the Django app package.
# Override by setting RF_MODEL_PATH to an absolute path in Django settings.
# ---------------------------------------------------------------------------
RF_MODEL_PATH = Path(__file__).parent / "ml_models" / "profile_matcher.pkl"


# ---------------------------------------------------------------------------
# 1. KDE Heatmap Helper
# ---------------------------------------------------------------------------

def compute_kde_heatmap(
    coordinates: list[tuple[float, float]],
    grid_size: int = 100,
    bandwidth: float = 0.05,
) -> dict[str, Any]:
    """
    Compute a Kernel Density Estimate over lat/lng coordinates and return
    a flat list of {lat, lng, intensity} dicts suitable for the frontend
    heatmap layer.

    Args:
        coordinates: List of (lat, lng) tuples.
        grid_size:   Number of grid cells per axis (higher → finer resolution).
        bandwidth:   KDE bandwidth in degrees (≈ 5 km at Zimbabwe's latitude).

    Returns:
        {'points': [{lat, lng, intensity}, …], 'max_intensity': float}
    """
    from sklearn.neighbors import KernelDensity

    if not coordinates:
        return {"points": [], "max_intensity": 0}

    coords = np.array(coordinates)

    # Fit KDE on raw lat/lng — bandwidth is in the same unit (degrees).
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords)

    # Build a regular grid spanning the bounding box of the data.
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lng_grid = np.linspace(lng_min, lng_max, grid_size)

    # meshgrid gives every (lat, lng) combination on the grid.
    grid_lat, grid_lng = np.meshgrid(lat_grid, lng_grid)
    grid_points = np.column_stack([grid_lat.ravel(), grid_lng.ravel()])

    # log_prob → exponentiate to get actual density.
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    # Normalise to [0, 1] for consistent frontend rendering.
    max_density = density.max() if density.max() > 0 else 1.0
    normalised = density / max_density

    # Only return cells above a small threshold to keep the payload small.
    threshold = 0.05
    mask = normalised > threshold
    points = [
        {"lat": float(grid_points[i, 0]),
         "lng": float(grid_points[i, 1]),
         "intensity": round(float(normalised[i]), 4)}
        for i in np.where(mask)[0]
    ]

    return {"points": points, "max_intensity": round(float(max_density), 6)}


# ---------------------------------------------------------------------------
# 2. Time Series Decomposition — Trend Analysis
# ---------------------------------------------------------------------------


def compute_time_series(
    df: pd.DataFrame,
    period: str = "weekly",
) -> dict[str, Any]:
    """
    Resample a crime incidents DataFrame by the requested period, run
    seasonal_decompose, and return components as JSON-serialisable lists.

    Args:
        df:     DataFrame with at least a 'timestamp' column and one row per
                incident. Must NOT be empty.
        period: 'daily' | 'weekly' | 'monthly'

    Returns:
        {
          'labels':   [...ISO date strings...],
          'observed': [...counts...],
          'trend':    [...floats or null...],
          'seasonal': [...floats...],
          'residual': [...floats or null...],
          'period_label': 'weekly',
          'total_incidents': int,
        }
    """
    from statsmodels.tsa.seasonal import seasonal_decompose  # local import

    resample_map = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}
    freq_rule = resample_map.get(period, "W-MON")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts = df.set_index("timestamp").resample(freq_rule).size()
    ts.index = ts.index.tz_localize(None)  # remove tz for statsmodels compatibility

    # Fill gaps with 0 so the time series has no missing steps.
    ts = ts.asfreq(freq_rule, fill_value=0)

    # seasonal_decompose requires at least 2 full cycles to be meaningful.
    min_periods = {"daily": 14, "weekly": 4, "monthly": 4}
    min_len = min_periods.get(period, 4)

    labels = ts.index.strftime("%Y-%m-%d").tolist()
    observed = ts.tolist()

    if len(ts) < min_len * 2:
        # Not enough data for decomposition — return observed only with a note.
        return {
            "labels": labels,
            "observed": observed,
            "trend": None,
            "seasonal": None,
            "residual": None,
            "period_label": period,
            "total_incidents": int(ts.sum()),
            "note": "Insufficient data for seasonal decomposition.",
        }

    model_type = "additive"
    decomposition = seasonal_decompose(
        ts, model=model_type,
        period=min_periods.get(period, 4),
        extrapolate_trend="freq",
    )

    def _to_list(series):
        """Convert a pandas Series to a list, replacing NaN with None for JSON."""
        return [None if np.isnan(v) else round(float(v), 4) for v in series]

    return {
        "labels": labels,
        "observed": observed,
        "trend": _to_list(decomposition.trend),
        "seasonal": _to_list(decomposition.seasonal),
        "residual": _to_list(decomposition.resid),
        "period_label": period,
        "total_incidents": int(ts.sum()),
    }


# ---------------------------------------------------------------------------
# 3. Random Forest Profile Matcher
# ---------------------------------------------------------------------------


class ProfileMatcher:
    """
    Encapsulates training and inference for the Random Forest crime-profile
    matching model.  The model is persisted to disk so the API view only
    needs to call find_similar() or load_and_predict().

    Feature set (X):
      - TF-IDF vectors from modus_operandi text
      - One-Hot: crime_type, time_of_day, day_of_week
      - One-Hot: weapon_used
    Target (y):
      - serial_group_label (groups assigned by analysts in the DB)
    """

    def __init__(self):
        self.rf = None
        self.tfidf = None
        self.crime_type_classes_: list = []
        self.tod_classes_: list = []
        self.dow_classes_: list = []
        self.weapon_classes_: list = []
        self.feature_cols_: list = []

    # ------------------------------------------------------------------
    # Training (offline — called by the train_profile_matcher management
    # command or the POST /api/zrp/ml/train/ endpoint).
    # ------------------------------------------------------------------

    def train(self, incidents_qs) -> dict:
        """
        Train (or re-train) the Random Forest on all labelled incidents.

        incidents_qs: Django QuerySet of CrimeIncident with non-empty
                      serial_group_label values.
        Returns: dict with training metrics.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score

        # Pull only the fields needed — avoids fetching heavy geometry columns.
        data = list(incidents_qs.values(
            "id", "modus_operandi", "crime_type__name",
            "time_of_day", "day_of_week", "weapon_used", "serial_group_label"
        ))

        if len(data) < 10:
            return {"error": "Need at least 10 labelled incidents to train."}

        df = pd.DataFrame(data)
        df = df[df["serial_group_label"].str.strip() != ""].copy()

        if df.empty:
            return {"error": "No incidents with serial_group_label found."}

        # ── TF-IDF on modus operandi free text ──────────────────────────────
        # max_features=300 keeps the vector space manageable; bigrams capture
        # multi-word MO patterns like "smash and grab".
        self.tfidf = TfidfVectorizer(
            max_features=300, ngram_range=(1, 2), stop_words="english"
        )
        mo_matrix = self.tfidf.fit_transform(
            df["modus_operandi"].fillna("")
        ).toarray()

        # ── One-Hot encode categorical features ─────────────────────────────
        # We store the classes seen during training so inference can replicate
        # the exact same column ordering.
        self.crime_type_classes_ = sorted(
            df["crime_type__name"].dropna().unique().tolist()
        )
        self.tod_classes_ = ["morning", "afternoon", "evening", "night"]
        self.dow_classes_ = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        ]
        self.weapon_classes_ = sorted(
            df["weapon_used"].fillna("").unique().tolist()
        )

        def ohe(series, classes):
            """Manually one-hot encode a Series given a fixed class list."""
            return np.array([[1 if val == c else 0 for c in classes]
                             for val in series])

        ct_ohe     = ohe(df["crime_type__name"].fillna(""), self.crime_type_classes_)
        tod_ohe    = ohe(df["time_of_day"].fillna(""),      self.tod_classes_)
        dow_ohe    = ohe(df["day_of_week"].fillna(""),      self.dow_classes_)
        weapon_ohe = ohe(df["weapon_used"].fillna(""),      self.weapon_classes_)

        # Stack all feature groups into a single 2-D feature matrix.
        X = np.hstack([mo_matrix, ct_ohe, tod_ohe, dow_ohe, weapon_ohe])
        y = df["serial_group_label"].values

        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
        # cross_val_score gives an unbiased accuracy estimate without a
        # separate hold-out split — important when data is limited.
        cv_scores = cross_val_score(
            self.rf, X, y,
            cv=min(3, len(df) // 5 or 1),
            scoring="accuracy",
        )
        self.rf.fit(X, y)

        # Persist the trained model to disk for later API inference.
        self._save()

        return {
            "status": "trained",
            "n_samples": len(df),
            "n_classes": len(self.rf.classes_),
            "classes": self.rf.classes_.tolist(),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std":  round(float(cv_scores.std()), 4),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        mo_text: str,
        crime_type_name: str,
        time_of_day: str,
        day_of_week: str,
        weapon_used: str,
    ) -> np.ndarray:
        """
        Reproduce the exact same feature engineering used during training so
        inference columns always line up with the trained model weights.
        Returns a (1, n_features) array ready to pass into self.rf.
        """
        # TF-IDF uses transform() — never fit_transform() — at inference time
        # so the vocabulary stays fixed to what was seen during training.
        mo_vec = self.tfidf.transform([mo_text]).toarray()

        def ohe(val, classes):
            """One-hot encode a single value against a fixed class list."""
            return np.array([[1 if val == c else 0 for c in classes]])

        ct_ohe     = ohe(crime_type_name, self.crime_type_classes_)
        tod_ohe    = ohe(time_of_day,     self.tod_classes_)
        dow_ohe    = ohe(day_of_week,     self.dow_classes_)
        wp_ohe     = ohe(weapon_used,     self.weapon_classes_)

        return np.hstack([mo_vec, ct_ohe, tod_ohe, dow_ohe, wp_ohe])

    def predict(
        self,
        mo_text: str,
        crime_type_name: str,
        time_of_day: str,
        day_of_week: str,
        weapon_used: str,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rank serial-group labels by predicted probability.

        Returns a list of {group_label, probability} dicts sorted descending,
        filtered to only groups with probability > 1 %.
        """
        if self.rf is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._build_feature_vector(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used
        )
        probas  = self.rf.predict_proba(X)[0]
        classes = self.rf.classes_

        ranked = sorted(
            zip(classes, probas), key=lambda x: x[1], reverse=True
        )[:top_n]
        return [
            {"group_label": cls, "probability": round(float(prob), 4)}
            for cls, prob in ranked
            if prob > 0.01
        ]

    # ------------------------------------------------------------------
    # find_similar  (FIX — this method was called in views.py but never
    # implemented; adding it here resolves the AttributeError).
    # ------------------------------------------------------------------

    def find_similar(
        self,
        incident,                    # CrimeIncident Django model instance
        top_n: int = 5,
    ) -> list[int]:
        """
        Given a single CrimeIncident ORM instance, return the PKs of the
        top_n most similar incidents in the database.

        Strategy
        --------
        1. Load the persisted model (must be trained first).
        2. Build a feature vector for the query incident.
        3. Compute the cosine similarity between the query vector and every
           stored training-sample vector that the Random Forest's underlying
           TF-IDF + feature space can represent.
        4. Because we don't store all training vectors at model-save time,
           we fall back to a *label-based* approach: predict the serial group
           label for the query incident, then fetch the top_n incidents from
           the database that share that label — excluding the query itself.

        This is the most pragmatic approach given the existing model design.
        If the model has not been trained yet, a FileNotFoundError is raised
        and the calling view converts it to a 503 response.

        Args:
            incident: A CrimeIncident instance (needs modus_operandi,
                      crime_type, time_of_day, day_of_week, weapon_used,
                      serial_group_label).
            top_n:    How many similar incident PKs to return.

        Returns:
            List of integer PKs (excludes the query incident's own PK).
        """
        # ── Import here to avoid circular imports at module level ────────────
        from .models import CrimeIncident  # type: ignore

        # ── Load the persisted model from disk ───────────────────────────────
        # Raises FileNotFoundError if train() has never been run.
        loaded = ProfileMatcher.load()

        # ── Extract feature fields from the ORM instance ─────────────────────
        mo_text        = incident.modus_operandi or ""
        crime_type_name = (
            incident.crime_type.name if incident.crime_type else ""
        )
        time_of_day  = incident.time_of_day  or ""
        day_of_week  = incident.day_of_week  or ""
        weapon_used  = incident.weapon_used  or ""

        # ── Build the query feature vector ────────────────────────────────────
        X_query = loaded._build_feature_vector(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used
        )

        # ── Predict the most likely serial group label(s) ────────────────────
        # We use predict_proba to rank all groups and take the top result.
        probas  = loaded.rf.predict_proba(X_query)[0]
        classes = loaded.rf.classes_

        # Sort groups by probability descending; we try the best-matching
        # group first, then fall back to lower-ranked groups if needed.
        ranked_groups = [
            cls for cls, _ in sorted(
                zip(classes, probas), key=lambda x: x[1], reverse=True
            )
        ]

        # ── Fetch candidates from the DB that share the predicted label ───────
        # Try each ranked group until we collect enough candidates.
        candidate_ids: list[int] = []
        for group_label in ranked_groups:
            if len(candidate_ids) >= top_n:
                break
            qs = (
                CrimeIncident.objects
                .filter(serial_group_label=group_label)
                .exclude(pk=incident.pk)        # never return the query itself
                .values_list("id", flat=True)
                [:top_n]
            )
            for pk in qs:
                if pk not in candidate_ids:
                    candidate_ids.append(pk)

        # ── Fallback: if the model has no usable labels, use cosine similarity
        # on the TF-IDF MO vectors alone (unsupervised, always available).
        if not candidate_ids:
            logger.warning(
                "find_similar: no label-based matches found for incident %s; "
                "falling back to TF-IDF cosine similarity.",
                incident.pk,
            )
            candidate_ids = loaded._find_similar_by_cosine(
                incident, top_n=top_n
            )

        return candidate_ids[:top_n]

    def _find_similar_by_cosine(
        self,
        query_incident,
        top_n: int = 5,
    ) -> list[int]:
        """
        Pure TF-IDF cosine-similarity fallback used when the supervised model
        has no group labels to match against.

        Fetches all incidents that have a non-empty modus_operandi, encodes
        them with the trained TF-IDF vectoriser, and returns the PKs of the
        top_n most similar ones to the query.

        This is an O(N) scan — acceptable for datasets up to ~100 k rows;
        for larger datasets consider an ANN index (e.g. FAISS).
        """
        from .models import CrimeIncident  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        # Pull all candidate incidents' PK + MO text in one DB query.
        candidates = list(
            CrimeIncident.objects
            .exclude(pk=query_incident.pk)
            .exclude(modus_operandi="")
            .values("id", "modus_operandi")
        )

        if not candidates:
            return []

        # TF-IDF encode query and all candidates using the *training* vocabulary.
        query_vec = self.tfidf.transform(
            [query_incident.modus_operandi or ""]
        )
        cand_texts = [c["modus_operandi"] or "" for c in candidates]
        cand_vecs  = self.tfidf.transform(cand_texts)

        # cosine_similarity returns a (1, N) matrix; flatten to 1-D.
        sims = cos_sim(query_vec, cand_vecs).flatten()

        # Pick the indices of the top_n highest similarity scores.
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [candidates[i]["id"] for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Serialise the entire ProfileMatcher instance to disk with pickle."""
        RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RF_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("ProfileMatcher saved to %s", RF_MODEL_PATH)

    @classmethod
    def load(cls) -> "ProfileMatcher":
        """
        Load the persisted model from disk.
        Raises FileNotFoundError if train() has never been run.
        """
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Profile matcher model not found. "
                "Run: python manage.py train_profile_matcher"
            )
        with open(RF_MODEL_PATH, "rb") as f:
            instance = pickle.load(f)
        return instance

    @classmethod
    def load_and_predict(
        cls,
        mo_text: str,
        crime_type_name: str,
        time_of_day: str,
        day_of_week: str,
        weapon_used: str,
        top_n: int = 5,
    ) -> list[dict]:
        """Convenience class-method: load model then call predict()."""
        matcher = cls.load()
        return matcher.predict(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n
        )


# ---------------------------------------------------------------------------
# 4. Hotspot Summary Helper
# ---------------------------------------------------------------------------


def compute_hotspot_summary(
    coordinates: list[tuple[float, float]],
    crime_types: list[str],
    suburbs: list[str],
    eps_km: float = 0.5,
    min_samples: int = 3,
) -> list[dict]:
    """
    Uses DBSCAN spatial clustering to identify crime hotspot zones, then
    returns a summarised list sorted by incident count descending.

    Args:
        coordinates: List of (lat, lng) tuples.
        crime_types: Parallel list of crime type names.
        suburbs:     Parallel list of suburb strings.
        eps_km:      DBSCAN neighbourhood radius in km (default 500 m).
        min_samples: Minimum incidents needed to form a cluster core point.

    Returns:
        List of dicts: {area, suburb, incident_count, risk_level,
                        centre_lat, centre_lng}
    """
    from sklearn.cluster import DBSCAN

    if len(coordinates) < min_samples:
        return []

    coords = np.array(coordinates)

    # Convert km radius to radians for the haversine metric.
    # haversine metric expects [lat_rad, lng_rad]; earth radius ≈ 6371 km.
    coords_rad = np.radians(coords)
    eps_rad = eps_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # noise — not part of any cluster
        if label not in clusters:
            clusters[label] = {"indices": [], "crime_types": [], "suburbs": []}
        clusters[label]["indices"].append(idx)
        clusters[label]["crime_types"].append(crime_types[idx])
        clusters[label]["suburbs"].append(suburbs[idx])

    summaries = []
    for label, data in clusters.items():
        idxs = data["indices"]
        cluster_coords = coords[idxs]

        # Centroid of the cluster — simple arithmetic mean of lat/lng.
        centre_lat = float(cluster_coords[:, 0].mean())
        centre_lng = float(cluster_coords[:, 1].mean())

        count = len(idxs)

        # Risk level thresholds — can be tuned via Django settings.
        if count >= 20:
            risk = "Critical"
        elif count >= 10:
            risk = "High"
        elif count >= 5:
            risk = "Medium"
        else:
            risk = "Low"

        # Most common suburb name in this cluster.
        suburb_series = pd.Series(data["suburbs"])
        dominant_suburb = suburb_series.mode()[0] if not suburb_series.empty else ""

        # Most common crime type in this cluster.
        ct_series = pd.Series(data["crime_types"])
        dominant_ct = ct_series.mode()[0] if not ct_series.empty else ""

        summaries.append({
            "area":           dominant_ct,
            "suburb":         dominant_suburb,
            "incident_count": count,
            "risk_level":     risk,
            "centre_lat":     round(centre_lat, 6),
            "centre_lng":     round(centre_lng, 6),
        })

    return sorted(summaries, key=lambda x: x["incident_count"], reverse=True)
