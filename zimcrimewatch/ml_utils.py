"""
ZimCrimeWatch - ML & Analytics Utilities
===========================================
Contains all data-science logic called by the API views:
  1. Kernel Density Estimation  → heatmap data for Leaflet.js
  2. Time Series Decomposition  → trend / seasonal / residual data for Chart.js
  3. Random Forest Profile Matcher → link serial crimes by M.O. similarity
  4. Hotspot summary helper       → tabular hotspot list with counts and centre
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)
RF_MODEL_PATH = MODEL_DIR / "profile_matcher.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# 1. Kernel Density Estimation — Heatmap
# ---------------------------------------------------------------------------


def compute_kde_heatmap(
    coordinates: list[tuple[float, float]],
    bandwidth: float = 0.01,
    grid_size: int = 50,
    bounds: dict | None = None,
) -> list[list[float]]:
    """
    Run KDE over a list of (lat, lng) pairs and return a grid suitable
    for Leaflet.js heatmap plugin.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        bandwidth:   KDE bandwidth in degrees (≈1.1 km at 0.01°).
        grid_size:   Number of grid divisions per axis.
        bounds:      {'min_lat', 'max_lat', 'min_lng', 'max_lng'} or None
                     (auto-derived from data with a 0.05° padding).

    Returns:
        List of [latitude, longitude, intensity] triples, intensity in [0, 1].
    """
    from sklearn.neighbors import KernelDensity  # local import keeps startup fast

    if len(coordinates) < 2:
        # Not enough points for meaningful KDE
        return [[lat, lng, 1.0] for lat, lng in coordinates]

    coords_array = np.array(coordinates, dtype=float)
    lats, lngs = coords_array[:, 0], coords_array[:, 1]

    if bounds is None:
        pad = 0.05
        bounds = {
            "min_lat": lats.min() - pad,
            "max_lat": lats.max() + pad,
            "min_lng": lngs.min() - pad,
            "max_lng": lngs.max() + pad,
        }

    lat_grid = np.linspace(bounds["min_lat"], bounds["max_lat"], grid_size)
    lng_grid = np.linspace(bounds["min_lng"], bounds["max_lng"], grid_size)
    grid_lats, grid_lngs = np.meshgrid(lat_grid, lng_grid)
    grid_points = np.column_stack([grid_lats.ravel(), grid_lngs.ravel()])

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", metric="haversine")
    # haversine expects radians
    kde.fit(np.radians(coords_array))
    log_density = kde.score_samples(np.radians(grid_points))
    density = np.exp(log_density)

    # Normalise to [0, 1]
    density_min, density_max = density.min(), density.max()
    if density_max > density_min:
        density_norm = (density - density_min) / (density_max - density_min)
    else:
        density_norm = np.ones_like(density)

    # Filter out near-zero points to reduce payload size
    threshold = 0.05
    result = []
    for (lat, lng), intensity in zip(grid_points, density_norm):
        if intensity >= threshold:
            result.append([round(float(lat), 6), round(float(lng), 6), round(float(intensity), 4)])

    return result


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
    ts.index = ts.index.tz_localize(None)  # remove tz for statsmodels

    # Fill gaps with 0
    ts = ts.asfreq(freq_rule, fill_value=0)

    # seasonal_decompose needs at least 2 full cycles
    min_periods = {"daily": 14, "weekly": 4, "monthly": 4}
    min_len = min_periods.get(period, 4)

    labels = ts.index.strftime("%Y-%m-%d").tolist()
    observed = ts.tolist()

    if len(ts) < min_len * 2:
        # Not enough data for decomposition — return observed only
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
    decomposition = seasonal_decompose(ts, model=model_type, period=min_periods.get(period, 4), extrapolate_trend="freq")

    def _to_list(series):
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
    needs to call `load_and_predict()`.

    Feature set (X):
      - TF-IDF vectors from modus_operandi text
      - One-Hot: crime_type, time_of_day, day_of_week
      - Label-encoded: weapon_used
    Target (y):
      - serial_group_label (groups assigned by analysts in the DB)
    """

    def __init__(self):
        self.rf = None
        self.tfidf = None
        self.crime_type_classes_ = []
        self.tod_classes_ = []
        self.dow_classes_ = []
        self.weapon_classes_ = []
        self.feature_cols_ = []

    # ------------------------------------------------------------------
    # Training (offline / management command)
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
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score

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

        # TF-IDF on MO text
        self.tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
        mo_matrix = self.tfidf.fit_transform(df["modus_operandi"].fillna("")).toarray()

        # One-Hot encode categoricals manually (to be re-applicable at inference)
        self.crime_type_classes_ = sorted(df["crime_type__name"].dropna().unique().tolist())
        self.tod_classes_ = ["morning", "afternoon", "evening", "night"]
        self.dow_classes_ = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        self.weapon_classes_ = sorted(df["weapon_used"].fillna("").unique().tolist())

        def ohe(series, classes):
            return np.array([[1 if val == c else 0 for c in classes] for val in series])

        ct_ohe = ohe(df["crime_type__name"].fillna(""), self.crime_type_classes_)
        tod_ohe = ohe(df["time_of_day"].fillna(""), self.tod_classes_)
        dow_ohe = ohe(df["day_of_week"].fillna(""), self.dow_classes_)
        weapon_ohe = ohe(df["weapon_used"].fillna(""), self.weapon_classes_)

        X = np.hstack([mo_matrix, ct_ohe, tod_ohe, dow_ohe, weapon_ohe])
        y = df["serial_group_label"].values

        self.rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(self.rf, X, y, cv=min(3, len(df) // 5 or 1), scoring="accuracy")
        self.rf.fit(X, y)

        self._save()
        return {
            "status": "trained",
            "n_samples": len(df),
            "n_classes": len(self.rf.classes_),
            "classes": self.rf.classes_.tolist(),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
        }

    # ------------------------------------------------------------------
    # Inference (called by API view)
    # ------------------------------------------------------------------

    def _build_feature_vector(self, mo_text: str, crime_type_name: str,
                               time_of_day: str, day_of_week: str, weapon_used: str) -> np.ndarray:
        mo_vec = self.tfidf.transform([mo_text]).toarray()

        def ohe(val, classes):
            return np.array([[1 if val == c else 0 for c in classes]])

        ct_ohe = ohe(crime_type_name, self.crime_type_classes_)
        tod_ohe = ohe(time_of_day, self.tod_classes_)
        dow_ohe = ohe(day_of_week, self.dow_classes_)
        wp_ohe = ohe(weapon_used, self.weapon_classes_)

        return np.hstack([mo_vec, ct_ohe, tod_ohe, dow_ohe, wp_ohe])

    def predict(self, mo_text: str, crime_type_name: str, time_of_day: str,
                day_of_week: str, weapon_used: str, top_n: int = 5) -> list[dict]:
        """
        Returns a ranked list of {group_label, probability} dicts.
        """
        X = self._build_feature_vector(mo_text, crime_type_name, time_of_day, day_of_week, weapon_used)
        probas = self.rf.predict_proba(X)[0]
        classes = self.rf.classes_

        ranked = sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"group_label": cls, "probability": round(float(prob), 4)} for cls, prob in ranked if prob > 0.01]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with open(RF_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("ProfileMatcher saved to %s", RF_MODEL_PATH)

    @classmethod
    def load(cls) -> "ProfileMatcher":
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError("Profile matcher model not found. Run the train_profile_matcher management command first.")
        with open(RF_MODEL_PATH, "rb") as f:
            instance = pickle.load(f)
        return instance

    @classmethod
    def load_and_predict(cls, mo_text: str, crime_type_name: str, time_of_day: str,
                          day_of_week: str, weapon_used: str, top_n: int = 5) -> list[dict]:
        matcher = cls.load()
        return matcher.predict(mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n)


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
        eps_km:      DBSCAN neighbourhood radius in km.
        min_samples: Minimum incidents to form a cluster.

    Returns:
        List of dicts: {cluster_id, centre_lat, centre_lng, incident_count,
                        dominant_crime, top_suburbs, eps_km}
    """
    from sklearn.cluster import DBSCAN

    if len(coordinates) < min_samples:
        return []

    coords_array = np.radians(np.array(coordinates, dtype=float))
    # DBSCAN with haversine — eps in radians
    eps_rad = eps_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree", metric="haversine")
    labels = db.fit_predict(coords_array)

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue  # noise
        if label not in clusters:
            clusters[label] = {"lats": [], "lngs": [], "crime_types": [], "suburbs": []}
        clusters[label]["lats"].append(float(coordinates[i][0]))
        clusters[label]["lngs"].append(float(coordinates[i][1]))
        clusters[label]["crime_types"].append(crime_types[i])
        clusters[label]["suburbs"].append(suburbs[i])

    result = []
    for label, data in clusters.items():
        crime_type_counts = pd.Series(data["crime_types"]).value_counts()
        suburb_counts = pd.Series([s for s in data["suburbs"] if s]).value_counts()
        result.append({
            "cluster_id": int(label),
            "centre_lat": round(np.mean(data["lats"]), 6),
            "centre_lng": round(np.mean(data["lngs"]), 6),
            "incident_count": len(data["lats"]),
            "dominant_crime": crime_type_counts.index[0] if not crime_type_counts.empty else "Unknown",
            "top_suburbs": suburb_counts.index[:3].tolist(),
            "eps_km": eps_km,
        })

    result.sort(key=lambda x: x["incident_count"], reverse=True)
    return result
