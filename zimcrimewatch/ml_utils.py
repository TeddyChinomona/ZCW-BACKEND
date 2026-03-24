"""
ml_utils.py  —  ZimCrimeWatch Machine Learning Utilities
=========================================================

Provides four core ML helpers consumed by the API views:
  1. compute_kde_heatmap      — KDE density surface for the heatmap overlay
  2. compute_time_series      — trend / seasonal / residual decomposition
  3. ProfileMatcher           — Random Forest classifier for serial-group linkage
  4. compute_hotspot_summary  — DBSCAN spatial clustering of crime locations

PARAMETER CONTRACT FIX:
  compute_hotspot_summary() now accepts `eps_km` and `min_samples` keyword
  arguments so that HotspotView can use the adaptive-DBSCAN ladder without
  passing those kwargs to a function that previously didn't accept them.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Path to the persisted Random Forest model file.
# Lives inside the Django app folder alongside this module.
RF_MODEL_PATH = Path(__file__).parent / "ml_models" / "profile_matcher.pkl"


# =============================================================================
# 1. KDE Heatmap
# =============================================================================

def compute_kde_heatmap(coordinates: list, grid_size: int = 100,
                        bandwidth: float = 0.05) -> dict:
    """
    Convert a list of (lat, lng) crime locations into a density heatmap.

    Parameters
    ----------
    coordinates : list of (lat, lng) tuples
    grid_size   : number of grid cells per axis (higher = finer resolution)
    bandwidth   : KDE bandwidth in degrees (~0.05° ≈ 5 km near Harare)

    Returns
    -------
    dict  {"points": [{"lat":…, "lng":…, "intensity": 0-1}, …],
           "max_intensity": float}
    """
    from sklearn.neighbors import KernelDensity

    if not coordinates:
        return {"points": [], "max_intensity": 0}

    # Stack (lat, lng) pairs into a (N, 2) numpy array
    coords = np.array(coordinates)

    # Fit a Gaussian kernel to the crime locations.
    # bandwidth controls how wide the "bump" around each point is.
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords)

    # Build a uniform grid across the bounding box of the data
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lng_grid = np.linspace(lng_min, lng_max, grid_size)

    # meshgrid produces all (lat, lng) combinations on the grid
    grid_lat, grid_lng = np.meshgrid(lat_grid, lng_grid)
    grid_points = np.column_stack([grid_lat.ravel(), grid_lng.ravel()])

    # score_samples returns log-density; exponentiate to recover density
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    # Normalise to [0, 1] so intensity is consistent across different maps
    max_density = density.max() if density.max() > 0 else 1.0
    normalised = density / max_density

    # Drop cells below 5 % of peak to keep the response payload small
    threshold = 0.05
    mask = normalised > threshold

    points = [
        {
            "lat": float(grid_points[i, 0]),
            "lng": float(grid_points[i, 1]),
            "intensity": round(float(normalised[i]), 4),
        }
        for i in np.where(mask)[0]
    ]

    return {"points": points, "max_intensity": round(float(max_density), 6)}


# =============================================================================
# 2. Time Series Decomposition
# =============================================================================

# Map the short frequency codes sent by the serializer to pandas resample rules
# and human-readable labels.
FREQ_MAP = {
    # Short codes accepted by TimeSeriesRequestSerializer
    "D":  ("D",      "daily"),
    "W":  ("W-MON",  "weekly"),
    "M":  ("MS",     "monthly"),
    # Long names accepted by the management command / direct calls
    "daily":   ("D",      "daily"),
    "weekly":  ("W-MON",  "weekly"),
    "monthly": ("MS",     "monthly"),
}


def compute_time_series(df: pd.DataFrame, period: str = "W") -> dict:
    """
    Count crime incidents per time bucket and decompose the series into
    trend, seasonal, and residual components.

    Parameters
    ----------
    df     : DataFrame with at least a 'timestamp' column
    period : frequency code — 'D', 'W', 'M'  (or long-form equivalents)

    Returns
    -------
    dict with keys: labels, observed, trend, seasonal, residual,
                    period_label, total_incidents, [note]
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Resolve short / long frequency codes to pandas rule + display label
    rule, period_label = FREQ_MAP.get(period, ("W-MON", "weekly"))

    # Parse timestamps and set as the index for resampling
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts = df.set_index("timestamp").resample(rule).size()

    # Remove timezone info — statsmodels does not support tz-aware DatetimeIndex
    ts.index = ts.index.tz_localize(None)
    ts = ts.asfreq(rule, fill_value=0)   # fill gaps with 0 (no crimes)

    labels   = ts.index.strftime("%Y-%m-%d").tolist()
    observed = ts.tolist()

    # Minimum periods needed per frequency for a reliable decomposition
    min_needed = {"daily": 14, "weekly": 4, "monthly": 4}
    min_len    = min_needed.get(period_label, 4)

    if len(ts) < min_len * 2:
        return {
            "labels": labels,
            "observed": observed,
            "trend": None,
            "seasonal": None,
            "residual": None,
            "period_label": period_label,
            "total_incidents": int(ts.sum()),
            "note": "Not enough data for seasonal decomposition.",
        }

    # Additive decomposition: observed = trend + seasonal + residual
    decomposition = seasonal_decompose(
        ts,
        model="additive",
        period=min_len,
        extrapolate_trend="freq",  # fills NaN edges of the trend component
    )

    def _to_list(series: pd.Series) -> list:
        """Convert pandas Series → Python list, replacing NaN with None."""
        return [None if np.isnan(v) else round(float(v), 4) for v in series]

    return {
        "labels":           labels,
        "observed":         observed,
        "trend":            _to_list(decomposition.trend),
        "seasonal":         _to_list(decomposition.seasonal),
        "residual":         _to_list(decomposition.resid),
        "period_label":     period_label,
        "total_incidents":  int(ts.sum()),
    }


# =============================================================================
# 3. ProfileMatcher  — Random Forest serial-crime classifier
# =============================================================================

class ProfileMatcher:
    """
    Trains a Random Forest to link crimes into serial groups and, at
    inference time, finds the most similar incidents to a query case.

    Training features
    -----------------
    • TF-IDF vector from modus_operandi text (up to 300 features)
    • One-hot encodings: crime type, time of day, day of week, weapon used

    Usage
    -----
    Offline:   ProfileMatcher().train(labelled_queryset)
    Online:    ProfileMatcher.load().find_similar(incident, top_n=5)
    """

    def __init__(self):
        self.rf      = None   # trained RandomForestClassifier
        self.tfidf   = None   # fitted TfidfVectorizer

        # Vocabulary / class lists saved at training time so inference
        # always produces feature vectors with exactly the same shape
        self.crime_type_classes_: list = []
        self.tod_classes_: list        = []
        self.dow_classes_: list        = []
        self.weapon_classes_: list     = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, incidents_qs) -> dict:
        """
        Fit the Random Forest on labelled CrimeIncidents.

        Parameters
        ----------
        incidents_qs : QuerySet of CrimeIncident objects,
                       all having a non-empty serial_group_label

        Returns
        -------
        dict  {"status", "n_samples", "n_classes", "cv_accuracy_mean",
               "cv_accuracy_std"}  or  {"error": <message>}
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score

        # Pull only the columns we need from the DB to avoid loading
        # heavy geometry or binary fields unnecessarily
        data = list(incidents_qs.values(
            "id",
            "modus_operandi",
            "crime_type__name",
            "time_of_day",
            "day_of_week",
            "weapon_used",
            "serial_group_label",
        ))

        if len(data) < 10:
            return {"error": "Need at least 10 labelled incidents to train."}

        df = pd.DataFrame(data)
        df = df[df["serial_group_label"].str.strip() != ""].copy()

        if df.empty:
            return {"error": "No incidents with serial_group_label found after filtering."}

        # ── Feature 1: TF-IDF on modus operandi text ─────────────────────
        # max_features=300 keeps the vector small; ngram_range=(1,2) also
        # captures two-word phrases like "smash and grab".
        self.tfidf = TfidfVectorizer(
            max_features=300, ngram_range=(1, 2), stop_words="english"
        )
        # fit_transform learns the vocabulary AND encodes all training texts
        mo_matrix = self.tfidf.fit_transform(
            df["modus_operandi"].fillna("")
        ).toarray()

        # ── Feature 2: One-hot encode categorical columns ─────────────────
        # Store class lists so inference vectors have the same column count
        self.crime_type_classes_ = sorted(
            df["crime_type__name"].dropna().unique().tolist()
        )
        # Time-of-day and day-of-week have fixed, known categories
        self.tod_classes_ = ["morning", "afternoon", "evening", "night"]
        self.dow_classes_ = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]
        self.weapon_classes_ = sorted(
            df["weapon_used"].fillna("").unique().tolist()
        )

        def _ohe_series(series: pd.Series, classes: list) -> np.ndarray:
            """One-hot encode an entire column → (N, len(classes)) array."""
            return np.array(
                [[1 if val == c else 0 for c in classes] for val in series]
            )

        ct_ohe  = _ohe_series(df["crime_type__name"].fillna(""), self.crime_type_classes_)
        tod_ohe = _ohe_series(df["time_of_day"].fillna(""),       self.tod_classes_)
        dow_ohe = _ohe_series(df["day_of_week"].fillna(""),       self.dow_classes_)
        wp_ohe  = _ohe_series(df["weapon_used"].fillna(""),       self.weapon_classes_)

        # Concatenate all feature groups side-by-side into one wide matrix
        X = np.hstack([mo_matrix, ct_ohe, tod_ohe, dow_ohe, wp_ohe])
        y = df["serial_group_label"].values

        # ── Train the Random Forest ───────────────────────────────────────
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )

        # Cross-validate for an honest accuracy estimate before fitting on all data
        n_folds  = min(3, len(df) // 5 or 1)
        cv_scores = cross_val_score(self.rf, X, y, cv=n_folds, scoring="accuracy")

        # Final fit on the complete dataset
        self.rf.fit(X, y)
        self._save()

        return {
            "status":            "trained",
            "n_samples":         len(df),
            "n_classes":         len(self.rf.classes_),
            "classes":           self.rf.classes_.tolist(),
            "cv_accuracy_mean":  round(float(cv_scores.mean()), 4),
            "cv_accuracy_std":   round(float(cv_scores.std()), 4),
        }

    # ------------------------------------------------------------------
    # Feature vector construction (shared by predict() and find_similar())
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
        Build the same (1, n_features) feature vector used at training time.
        Uses transform() — NOT fit_transform() — so the training vocabulary
        is re-used unchanged at inference.
        """
        # TF-IDF encode the single MO text string using the training vocabulary
        mo_vec = self.tfidf.transform([mo_text]).toarray()

        def _ohe_single(value: str, classes: list) -> np.ndarray:
            """One-hot encode a single value → (1, len(classes)) array."""
            return np.array([[1 if value == c else 0 for c in classes]])

        ct_ohe  = _ohe_single(crime_type_name, self.crime_type_classes_)
        tod_ohe = _ohe_single(time_of_day,     self.tod_classes_)
        dow_ohe = _ohe_single(day_of_week,     self.dow_classes_)
        wp_ohe  = _ohe_single(weapon_used,     self.weapon_classes_)

        return np.hstack([mo_vec, ct_ohe, tod_ohe, dow_ohe, wp_ohe])

    # ------------------------------------------------------------------
    # Inference — predict serial group for a new crime description
    # ------------------------------------------------------------------

    def predict(
        self,
        mo_text: str,
        crime_type_name: str,
        time_of_day: str,
        day_of_week: str,
        weapon_used: str,
        top_n: int = 5,
    ) -> list:
        """
        Return the top_n most likely serial group labels and their probabilities.

        Returns
        -------
        list of {"group_label": str, "probability": float}
        """
        if self.rf is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = self._build_feature_vector(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used
        )

        # predict_proba returns shape (1, n_classes); [0] extracts the 1-D array
        probas  = self.rf.predict_proba(X)[0]
        classes = self.rf.classes_

        # Sort by probability descending, keep only top_n with > 1 % confidence
        ranked = sorted(zip(classes, probas), key=lambda p: p[1], reverse=True)
        return [
            {"group_label": lbl, "probability": round(float(prob), 4)}
            for lbl, prob in ranked[:top_n]
            if prob > 0.01
        ]

    # ------------------------------------------------------------------
    # Inference — find most similar incidents by TF-IDF cosine similarity
    # ------------------------------------------------------------------

    def find_similar(self, incident, top_n: int = 5) -> list:
        """
        Return PKs of the top_n incidents most similar to *incident* using
        cosine similarity on TF-IDF encoded modus operandi text.

        Parameters
        ----------
        incident : CrimeIncident ORM object (needs .pk and .modus_operandi)
        top_n    : how many similar incident PKs to return

        Returns
        -------
        list of integer PKs (excluding the query incident itself)
        """
        # Lazy import to avoid circular references at module load
        from .models import CrimeIncident
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        # The model must be loaded (tfidf fitted) before calling find_similar()
        if self.tfidf is None:
            instance = self.__class__.load()
            return instance.find_similar(incident, top_n=top_n)

        # Fetch all other incidents that have some MO text to compare against
        candidates = list(
            CrimeIncident.objects
            .exclude(pk=incident.pk)
            .exclude(modus_operandi="")
            .values("id", "modus_operandi")
        )

        if not candidates:
            return []

        # Encode the query incident's MO text using the TRAINING vocabulary
        query_vec = self.tfidf.transform([incident.modus_operandi or ""])

        # Encode all candidates in one batch call for efficiency
        cand_texts = [c["modus_operandi"] or "" for c in candidates]
        cand_vecs  = self.tfidf.transform(cand_texts)

        # cosine_similarity returns (1, N); flatten to 1-D for easy sorting
        similarities = cos_sim(query_vec, cand_vecs).flatten()

        # argsort ascending; reverse and slice to get top_n descending
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [candidates[i]["id"] for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Serialise the entire fitted ProfileMatcher to disk via pickle."""
        RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RF_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("ProfileMatcher saved → %s", RF_MODEL_PATH)

    @classmethod
    def load(cls) -> "ProfileMatcher":
        """
        Deserialise a previously saved ProfileMatcher from disk.

        Raises
        ------
        FileNotFoundError if train() has never been called.
        """
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Profile matcher model not found. "
                "Run: python manage.py train_profile_matcher  "
                "(or POST /api/zrp/ml/train/)"
            )
        with open(RF_MODEL_PATH, "rb") as f:
            instance = pickle.load(f)
        logger.info("ProfileMatcher loaded ← %s", RF_MODEL_PATH)
        return instance

    @classmethod
    def load_and_predict(
        cls, mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n=5
    ) -> list:
        """Convenience: load model from disk then call predict()."""
        return cls.load().predict(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n
        )


# =============================================================================
# 4. Hotspot Summary  — DBSCAN spatial clustering
# =============================================================================

def compute_hotspot_summary(
    coordinates: list,
    crime_types: list,
    suburbs: list,
    eps_km: float = 0.5,
    min_samples: int = 3,
) -> list:
    """
    Cluster crime locations into hotspot zones using DBSCAN and return a
    summary of each cluster.

    Parameters
    ----------
    coordinates  : list of (lat, lng) tuples — one per incident
    crime_types  : list of str — dominant crime type per incident
    suburbs      : list of str — suburb name per incident
    eps_km       : DBSCAN neighbourhood radius in kilometres (default 0.5 km)
    min_samples  : DBSCAN minimum points to form a cluster (default 3)

    Returns
    -------
    list of hotspot dicts, sorted by incident count descending.
    Each dict contains:
        cluster_id, centre_lat, centre_lng, radius_m, incident_count,
        risk_level, area (dominant crime type), suburb (dominant suburb)
    """
    from sklearn.cluster import DBSCAN

    if not coordinates:
        return []

    coords = np.array(coordinates)

    # Convert (lat, lng) from degrees to radians — required by the haversine metric
    coords_rad = np.radians(coords)

    # Convert eps from kilometres to radians using Earth radius 6371 km
    eps_rad = eps_km / 6371.0

    # Fit DBSCAN.
    # algorithm="ball_tree" + metric="haversine" gives proper spherical distances
    # so clusters respect real-world geography rather than flat Euclidean distance.
    dbscan = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )
    cluster_labels = dbscan.fit_predict(coords_rad)

    hotspots = []
    unique_labels = sorted(set(cluster_labels))

    for label in unique_labels:
        if label == -1:
            # DBSCAN uses -1 for "noise" points that don't belong to any cluster
            continue

        # Boolean mask: True for every crime assigned to this cluster
        mask = cluster_labels == label
        cluster_coords = coords[mask]

        # Centroid = mean of latitudes and longitudes within the cluster
        centre_lat = float(cluster_coords[:, 0].mean())
        centre_lng = float(cluster_coords[:, 1].mean())

        # Radius = maximum distance from centroid to any member (degrees → metres)
        dists_deg = np.sqrt(
            (cluster_coords[:, 0] - centre_lat) ** 2 +
            (cluster_coords[:, 1] - centre_lng) ** 2
        )
        radius_m = float(dists_deg.max() * 111_000)  # 1° ≈ 111 km

        # Count incidents per crime type within the cluster
        cluster_crime_types = [
            crime_types[i] for i, flag in enumerate(mask) if flag
        ]
        ct_counts: dict = {}
        for ct in cluster_crime_types:
            ct_counts[ct] = ct_counts.get(ct, 0) + 1

        # Dominant crime type (most frequent)
        dominant_ct = max(ct_counts, key=ct_counts.get) if ct_counts else "Unknown"

        # Dominant suburb (most frequent non-empty)
        cluster_suburbs = [
            suburbs[i] for i, flag in enumerate(mask) if flag and suburbs[i]
        ]
        if cluster_suburbs:
            sb_counts: dict = {}
            for s in cluster_suburbs:
                sb_counts[s] = sb_counts.get(s, 0) + 1
            dominant_suburb = max(sb_counts, key=sb_counts.get)
        else:
            dominant_suburb = "Unknown"

        # Assign a risk level based on cluster size.
        # These thresholds can be tuned to ZRP operational needs.
        count = int(mask.sum())
        if count >= 20:
            risk_level = "Critical"
        elif count >= 10:
            risk_level = "High"
        elif count >= 5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        hotspots.append({
            "cluster_id":     int(label),
            # "area" carries the dominant crime type label
            # (matches the column name the frontend Analytics component reads)
            "area":           dominant_ct,
            # "suburb" carries the dominant location name
            "suburb":         dominant_suburb,
            "incident_count": count,
            "risk_level":     risk_level,
            # Centroid coordinates — used by the Leaflet map in Analytics.jsx
            "centre_lat":     round(centre_lat, 6),
            "centre_lng":     round(centre_lng, 6),
            "radius_m":       round(radius_m, 1),
            "crime_type_breakdown": ct_counts,
        })

    # Largest clusters first so the table rows are pre-sorted
    hotspots.sort(key=lambda h: h["incident_count"], reverse=True)
    return hotspots