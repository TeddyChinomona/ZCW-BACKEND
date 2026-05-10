"""
ml_utils.py  —  ZimCrimeWatch Machine Learning Utilities
=========================================================

What this file provides:
  1. compute_kde_heatmap    — turns crime coordinates into a density heatmap
  2. compute_time_series    — breaks crime counts into trend/seasonal/residual
  3. ProfileMatcher         — Random Forest classifier that groups crimes by similarity
       .train()             — trains the model from the database
       .find_similar()      — finds crimes similar to a given incident
       .predict()           — ranks possible serial-group labels for a new incident
       .load() / _save()    — saves/loads the model to/from disk
  4. compute_hotspot_summary — clusters crime locations using DBSCAN

Bug fixes in this version
--------------------------
  • compute_hotspot_summary now accepts `eps_km` and `min_samples` keyword
    arguments so the adaptive-DBSCAN loop in HotspotView works correctly.
  • compute_time_series now maps the single-letter freq codes ("D", "W", "M")
    that TimeSeriesRequestSerializer sends, in addition to the long-form names.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Where the trained Random Forest model gets saved on disk.
# The folder "ml_models/" lives inside the Django app package folder.
RF_MODEL_PATH = Path(__file__).parent / "ml_models" / "profile_matcher.pkl"


# =============================================================================
# 1. KDE Heatmap  —  "How dense are crimes in each area of the map?"
# =============================================================================

# def compute_kde_heatmap(coordinates, grid_size=25, bandwidth=0.01):
#     """
#     Takes a list of (latitude, longitude) crime locations and returns
#     a grid of density values for drawing a heatmap on the frontend map.

#     How it works:
#       - We fit a Kernel Density Estimator (KDE) to the crime coordinates.
#         Think of KDE as placing a small "bump" at each crime location and
#         then adding all the bumps together.  Where many crimes cluster, the
#         bumps add up to a high peak.  Sparse areas stay low.
#       - We then evaluate the density on a regular grid of points across
#         the bounding box of the data.
#       - We normalise the density to a 0-1 scale so the frontend always gets
#         consistent intensity values regardless of the number of crimes.

#     Parameters
#     ----------
#     coordinates : list of (lat, lng) tuples
#     grid_size   : how many grid cells per axis (higher = finer detail)
#     bandwidth   : controls how wide each "bump" is, in degrees
#                   (~0.05° ≈ 5 km around Harare's latitude)

#     Returns
#     -------
#     {
#       "points": [ {"lat": ..., "lng": ..., "intensity": 0.0-1.0}, ... ],
#       "max_intensity": float   # raw peak density before normalisation
#     }
#     """
#     from sklearn.neighbors import KernelDensity

#     # Nothing to do if there are no coordinates
#     if not coordinates:
#         return {"points": [], "max_intensity": 0}

#     # Convert list of tuples to a 2D numpy array  shape: (N, 2)
#     coords = np.array(coordinates)

#     # Fit the KDE to the crime locations.
#     # KernelDensity learns where the data is dense vs sparse.
#     kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
#     kde.fit(coords)

#     # Build a regular grid that covers the entire dataset area.
#     lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
#     lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()

#     # linspace creates evenly-spaced points between the min and max.
#     lat_grid = np.linspace(lat_min, lat_max, grid_size)
#     lng_grid = np.linspace(lng_min, lng_max, grid_size)

#     # meshgrid turns two 1-D arrays into all (lat, lng) combinations on the
#     # grid.  grid_lat and grid_lng are both (grid_size × grid_size) matrices.
#     grid_lat, grid_lng = np.meshgrid(lat_grid, lng_grid)

#     # Flatten and zip the two matrices into a list of (lat, lng) points so we
#     # can score all grid cells in one call  shape: (grid_size², 2)
#     grid_points = np.column_stack([grid_lat.ravel(), grid_lng.ravel()])

#     # ---- Score every grid cell with the KDE --------------------------------
#     # KernelDensity.score_samples() returns log(density) for numerical
#     # stability.  We exponentiate to get the actual density value.
#     log_density = kde.score_samples(grid_points)
#     density     = np.exp(log_density)

#     # ---- Normalise density to the range [0, 1] ----------------------------
#     max_density = density.max() if density.max() > 0 else 1.0
#     normalised  = density / max_density

#     # ---- Filter out near-zero cells to keep the API response small --------
#     # Any cell with less than 5% of peak intensity is not worth sending.
#     threshold = 0.05
#     mask      = normalised > threshold

#     points = [
#         {
#             "lat":       float(grid_points[i, 0]),
#             "lng":       float(grid_points[i, 1]),
#             "intensity": round(float(normalised[i]), 4),
#         }
#         for i in np.where(mask)[0]
#     ]

#     return {"points": points, "max_intensity": round(float(max_density), 6)}

import numpy as np
from sklearn.neighbors import KernelDensity

def compute_kde_heatmap(coordinates, grid_size=100, bandwidth_km=1.5):
    if not coordinates:
        return {"points": [], "max_intensity": 0}
    logger.warning(coordinates)
    coords = np.array(coordinates)

    # 1. Convert degrees to radians for Haversine metric
    coords_rad = np.radians(coords)
    
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0 
    
    # Bandwidth must also be converted to radians
    bandwidth_rad = bandwidth_km / EARTH_RADIUS_KM

    # 2. Fit KDE using spherical distance
    kde = KernelDensity(
        bandwidth=bandwidth_rad, 
        kernel="gaussian", 
        metric="haversine"
    )
    kde.fit(coords_rad)

    # 3. Build the grid
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lng_grid = np.linspace(lng_min, lng_max, grid_size)
    grid_lat, grid_lng = np.meshgrid(lat_grid, lng_grid)

    grid_points = np.column_stack([grid_lat.ravel(), grid_lng.ravel()])
    
    # Convert grid points to radians before scoring
    grid_points_rad = np.radians(grid_points)

    # 4. Score and exponentiate
    log_density = kde.score_samples(grid_points_rad)
    density = np.exp(log_density)

    # 5. Normalise
    max_density = density.max() if density.max() > 0 else 1.0
    normalised = density / max_density

    # Filter and construct payload (keep threshold low if using high grid_size)
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
    logger.critical(points)
    return {"points": points, "max_intensity": round(float(max_density), 6)}

# =============================================================================
# 2. Time Series Decomposition  —  "What are the trends in crime over time?"
# =============================================================================

# ── Frequency code mapping ────────────────────────────────────────────────────
# TimeSeriesRequestSerializer sends single-letter codes ("D", "W", "M").
# The old code only accepted long-form names and always fell through to the
# default.  This combined map handles BOTH forms so nothing is ever silently
# ignored.
_FREQ_MAP = {
    # Long-form names (legacy / direct calls)
    "daily":   ("D",       14),   # (pandas resample code, min periods for decompose)
    "weekly":  ("W-MON",    4),
    "monthly": ("MS",       4),
    # Short codes sent by the serializer
    "D":       ("D",       14),
    "W":       ("W-MON",    4),
    "M":       ("MS",       4),
}


def compute_time_series(df, period="W"):
    """
    Counts crime incidents per time period and decomposes the count series
    into three components:
      - trend    : long-term direction (going up or down overall?)
      - seasonal : repeating patterns (e.g. more crime on weekends)
      - residual : what remains after trend and seasonal are removed (noise)

    This uses the "additive" model: observed = trend + seasonal + residual

    Parameters
    ----------
    df     : DataFrame with at least a 'timestamp' column (one row per crime)
    period : frequency code — accepts "D"/"W"/"M" (from serializer) OR
             "daily"/"weekly"/"monthly" (long-form).  Defaults to "W".

    Returns
    -------
    dict with lists for each component, suitable for JSON serialisation.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Look up the pandas resample code and minimum-periods threshold.
    # .upper() normalises "w" → "W" etc. just in case.
    freq_rule, min_periods = _FREQ_MAP.get(
        period,
        _FREQ_MAP.get(str(period).upper(), ("W-MON", 4)),
    )

    # Parse the timestamp column and set it as the DataFrame index.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # resample().size() counts how many crimes fall into each time bucket.
    ts = df.set_index("timestamp").resample(freq_rule).size()

    # Remove timezone info — statsmodels does not support timezone-aware indexes.
    ts.index = ts.index.tz_localize(None)

    # Fill any missing time periods with 0 (no crimes that period).
    ts = ts.asfreq(freq_rule, fill_value=0)

    labels   = ts.index.strftime("%Y-%m-%d").tolist()
    observed = ts.tolist()

    # seasonal_decompose needs at least 2 full cycles of data to work.
    if len(ts) < min_periods * 2:
        return {
            "labels":           labels,
            "observed":         observed,
            "trend":            None,
            "seasonal":         None,
            "residual":         None,
            "period_label":     period,
            "total_incidents":  int(ts.sum()),
            "note":             "Not enough data for seasonal decomposition.",
        }

    decomposition = seasonal_decompose(
        ts,
        model="additive",
        period=min_periods,
        extrapolate_trend="freq",   # fill NaN at the edges of the trend component
    )

    def series_to_list(series):
        """
        Convert a pandas Series to a plain Python list.
        NaN values (at the start/end of trend and residual) become None so
        they serialise cleanly as JSON null.
        """
        return [None if np.isnan(v) else round(float(v), 4) for v in series]

    logger.success(f"Residual data: {series_to_list(decomposition.resid)}")
    logger.success(f"Seasonal data: {series_to_list(decomposition.seasonal)}")
    logger.success(f"Trend data: {series_to_list(decomposition.trend)}")
    return {
        "labels":          labels,
        "observed":        observed,
        "trend":           series_to_list(decomposition.trend),
        "seasonal":        series_to_list(decomposition.seasonal),
        "residual":        series_to_list(decomposition.resid),
        "period_label":    period,
        "total_incidents": int(ts.sum()),
    }


# =============================================================================
# 3. ProfileMatcher  —  "Which crimes were likely committed by the same person?"
# =============================================================================

class ProfileMatcher:
    """
    A Random Forest classifier that learns to match crimes into serial groups.

    What it does:
      - During training: learns from crimes that analysts have labelled with a
        'serial_group_label' (e.g. "GROUP_A", "GROUP_B").
      - During inference: given a new crime, predicts which group it most
        likely belongs to, or finds the most textually-similar crimes.

    Features used:
      - TF-IDF text vector from the modus_operandi field
      - One-hot encoding of: crime type, time of day, day of week, weapon used
    """

    def __init__(self):
        # The trained Random Forest classifier (set after .train() is called)
        self.rf = None

        # TF-IDF vectoriser fitted on modus operandi text during training.
        # Saved so we can apply the same vocabulary at inference time.
        self.tfidf = None

        # The unique values seen during training for each categorical column.
        # Saved so one-hot encoding at inference always has the same layout.
        self.crime_type_classes_ = []
        self.tod_classes_        = []   # time-of-day categories
        self.dow_classes_        = []   # day-of-week categories
        self.weapon_classes_     = []

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self, incidents_qs):
        """
        Train the Random Forest on all labelled crime incidents.

        Parameters
        ----------
        incidents_qs : Django QuerySet of CrimeIncident objects that all have a
                       non-empty serial_group_label.

        Returns
        -------
        dict with accuracy metrics, e.g. {"cv_accuracy_mean": 0.87, ...}
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score

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
            return {"error": "No incidents with serial_group_label found."}

        # ---- Feature 1: TF-IDF on modus operandi free text -----------------
        # TF-IDF converts each text description into a numeric vector.
        # Words rare across all descriptions (and thus more distinctive) get a
        # higher weight.  max_features=300 keeps the vector compact;
        # ngram_range=(1,2) also captures two-word phrases like "smash and grab".
        self.tfidf = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words="english",
        )
        mo_matrix = self.tfidf.fit_transform(
            df["modus_operandi"].fillna("")
        ).toarray()  # .toarray() converts sparse matrix to dense numpy array

        # ---- Feature 2: One-hot encode categorical columns ------------------
        # One-hot encoding turns a category like "morning" into a binary
        # column: [1, 0, 0, 0] means "morning", [0, 1, 0, 0] means
        # "afternoon", etc.
        self.crime_type_classes_ = sorted(
            df["crime_type__name"].dropna().unique().tolist()
        )
        self.tod_classes_    = ["morning", "afternoon", "evening", "night"]
        self.dow_classes_    = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]
        self.weapon_classes_ = sorted(
            df["weapon_used"].fillna("").unique().tolist()
        )

        def one_hot_encode_series(series, classes):
            """
            Encode a whole pandas Series as a one-hot matrix.
            Returns a 2-D numpy array of shape (len(series), len(classes)).
            Each row has a 1 in the column matching its value, 0s elsewhere.
            """
            return np.array(
                [[1 if val == c else 0 for c in classes] for val in series]
            )

        ct_ohe     = one_hot_encode_series(df["crime_type__name"].fillna(""), self.crime_type_classes_)
        tod_ohe    = one_hot_encode_series(df["time_of_day"].fillna(""),       self.tod_classes_)
        dow_ohe    = one_hot_encode_series(df["day_of_week"].fillna(""),       self.dow_classes_)
        weapon_ohe = one_hot_encode_series(df["weapon_used"].fillna(""),       self.weapon_classes_)

        # ---- Combine all features into one big matrix ----------------------
        # np.hstack glues the matrices side by side:
        # [TF-IDF columns | crime_type columns | time_of_day columns | ...]
        X = np.hstack([mo_matrix, ct_ohe, tod_ohe, dow_ohe, weapon_ohe])
        y = df["serial_group_label"].values

        # ---- Train the Random Forest ----------------------------------------
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,          # use all available CPU cores
        )

        # cross_val_score splits the data into folds, trains on each fold,
        # and tests on the remaining fold to get an honest accuracy estimate
        # without needing a separate test set.
        n_folds   = min(3, len(df) // 5 or 1)
        cv_scores = cross_val_score(self.rf, X, y, cv=n_folds, scoring="accuracy")

        # Now train on ALL the data (cross-val was just for measuring accuracy).
        self.rf.fit(X, y)
        self._save()

        return {
            "status":             "trained",
            "n_samples":          len(df),
            "n_classes":          len(self.rf.classes_),
            "classes":            self.rf.classes_.tolist(),
            "cv_accuracy_mean":   round(float(cv_scores.mean()), 4),
            "cv_accuracy_std":    round(float(cv_scores.std()), 4),
        }

    # -------------------------------------------------------------------------
    # Feature building  —  used by both predict() and find_similar()
    # -------------------------------------------------------------------------

    def _build_feature_vector(
        self, mo_text, crime_type_name, time_of_day, day_of_week, weapon_used
    ):
        """
        Turn a single crime's attributes into the same numeric feature vector
        format used during training.

        This MUST mirror the feature engineering in train() exactly — calling
        transform() (not fit_transform()) so we reuse the learned vocabulary.

        Returns a (1, n_features) numpy array.
        """
        mo_vec = self.tfidf.transform([mo_text]).toarray()

        def one_hot_single(value, classes):
            """
            Encode ONE value as a one-hot vector of shape (1, len(classes)).
            Example: one_hot_single("morning", ["morning","afternoon","evening","night"])
                     → [[1, 0, 0, 0]]
            """
            return np.array([[1 if value == c else 0 for c in classes]])

        ct_ohe  = one_hot_single(crime_type_name, self.crime_type_classes_)
        tod_ohe = one_hot_single(time_of_day,     self.tod_classes_)
        dow_ohe = one_hot_single(day_of_week,     self.dow_classes_)
        wp_ohe  = one_hot_single(weapon_used,     self.weapon_classes_)

        # Glue all feature groups into one row vector  shape: (1, total_features)
        return np.hstack([mo_vec, ct_ohe, tod_ohe, dow_ohe, wp_ohe])

    # -------------------------------------------------------------------------
    # Inference: predict serial group labels for a new crime
    # -------------------------------------------------------------------------

    def predict(
        self, mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n=5
    ):
        """
        Given attributes of a crime, return the most likely serial group labels
        ranked by probability.

        Returns a list like:
          [{"group_label": "GROUP_A", "probability": 0.72}, ...]
        Only groups with probability > 1% are included.
        """
        if self.rf is None:
            raise RuntimeError("Model not trained yet.  Call train() first.")

        X      = self._build_feature_vector(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used
        )

        # predict_proba() returns a probability for each known group label.
        # probas shape: (1, n_classes) → [0] gives the 1-D array.
        probas  = self.rf.predict_proba(X)[0]
        classes = self.rf.classes_

        ranked = sorted(zip(classes, probas), key=lambda pair: pair[1], reverse=True)

        return [
            {"group_label": label, "probability": round(float(prob), 4)}
            for label, prob in ranked[:top_n]
            if prob > 0.01
        ]

    # -------------------------------------------------------------------------
    # Inference: find similar incidents to a given one
    # -------------------------------------------------------------------------

    def find_similar(self, incident, top_n=5):
        """
        Given a CrimeIncident database object, find the PKs of the top_n most
        textually similar incidents using TF-IDF cosine similarity on the
        modus_operandi text.

        Cosine similarity measures how "close" two text vectors are in
        direction (ignoring length), so two crimes described similarly score
        high regardless of text length.

        Parameters
        ----------
        incident : a CrimeIncident Django ORM instance
        top_n    : how many similar incident PKs to return

        Returns
        -------
        list of integer PKs (excluding the query incident itself)
        """
        from .models import CrimeIncident
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        if self.tfidf is None:
            raise RuntimeError(
                "TF-IDF vectoriser not available.  "
                "Run train_profile_matcher management command first."
            )

        candidates = list(
            CrimeIncident.objects
            .exclude(pk=incident.pk)
            .exclude(modus_operandi="")
            .values("id", "modus_operandi")
        )

        if not candidates:
            return []

        # Encode the query crime's MO text using the TRAINING vocabulary
        query_vec = self.tfidf.transform([incident.modus_operandi or ""])

        # Encode all candidate MO texts in one batch call (efficient)
        candidate_texts = [c["modus_operandi"] or "" for c in candidates]
        candidate_vecs  = self.tfidf.transform(candidate_texts)

        # cosine_similarity returns shape (1, N); .flatten() gives a 1-D array
        similarities = cos_sim(query_vec, candidate_vecs).flatten()

        # argsort ascending → [::-1] reverses to descending → [:top_n] keeps top
        top_indices = np.argsort(similarities)[::-1][:top_n]

        return [candidates[i]["id"] for i in top_indices]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        """Serialise this ProfileMatcher object to disk using pickle."""
        RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RF_MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("ProfileMatcher saved to %s", RF_MODEL_PATH)

    @classmethod
    def load(cls):
        """
        Load a previously saved ProfileMatcher from disk.
        Raises FileNotFoundError if train() has never been run.
        """
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Profile matcher model not found.  "
                "Run: python manage.py train_profile_matcher"
            )
        with open(RF_MODEL_PATH, "rb") as f:
            instance = pickle.load(f)
        logger.info("ProfileMatcher loaded from %s", RF_MODEL_PATH)
        return instance

    @classmethod
    def load_and_predict(
        cls, mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n=5
    ):
        """Convenience: load the model from disk, then call predict()."""
        matcher = cls.load()
        return matcher.predict(
            mo_text, crime_type_name, time_of_day, day_of_week, weapon_used, top_n
        )


# =============================================================================
# 4. Hotspot Summary  —  "Where are the crime hotspots?"
# =============================================================================

def compute_hotspot_summary(
    coordinates,
    crime_types,
    suburbs,
    eps_km=0.5,
    min_samples=3,
):
    """
    Groups crime locations into spatial clusters (hotspots) using DBSCAN,
    then summarises each cluster with a centroid, radius, and crime breakdown.

    Parameters
    ----------
    coordinates  : list of (lat, lng) tuples
    crime_types  : list of crime type strings, same length as coordinates
    suburbs      : list of suburb strings, same length as coordinates
    eps_km       : maximum radius of a DBSCAN neighbourhood in kilometres.
                   Default 0.5 km (500 m).  HotspotView passes progressively
                   larger values when no clusters are found at the tighter radius.
    min_samples  : minimum number of crimes required to form a cluster.
                   Default 3.  HotspotView reduces this to 2 on the looser passes.

    DBSCAN notes:
      - Groups points that are close to each other into clusters.
      - Points that are far from any cluster are labelled "noise" (label = -1).
      - Unlike k-means, you do NOT need to specify the number of clusters.

    Returns
    -------
    list of hotspot dicts, each containing centroid, radius, count, etc.
    """
    from sklearn.cluster import DBSCAN

    if not coordinates:
        return []

    coords = np.array(coordinates)

    # ---- Convert degrees to radians for the Haversine distance metric ------
    # Haversine measures real-world distance on the Earth's curved surface.
    # DBSCAN needs radians when metric="haversine".
    coords_rad = np.radians(coords)

    # ---- Convert the km radius to radians for DBSCAN -----------------------
    # Earth radius ≈ 6371 km.  eps_km / 6371 gives the radian equivalent.
    eps_rad = eps_km / 6371.0

    dbscan = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",     # proper spherical distance
    )
    cluster_labels = dbscan.fit_predict(coords_rad)

    # ---- Summarise each cluster ---------------------------------------------
    hotspots      = []
    unique_labels = set(cluster_labels)

    for label in unique_labels:
        # Label -1 means "noise" — crimes too isolated to form a hotspot
        if label == -1:
            continue

        # Boolean mask: True for every crime that belongs to this cluster
        mask           = cluster_labels == label
        cluster_coords = coords[mask]

        # Centroid = average lat and lng of all crimes in the cluster
        centroid_lat = float(cluster_coords[:, 0].mean())
        centroid_lng = float(cluster_coords[:, 1].mean())

        # Radius = furthest Euclidean distance (in degrees) from centroid,
        # converted to approximate metres (1 degree ≈ 111 km)
        distances_from_centroid = np.sqrt(
            (cluster_coords[:, 0] - centroid_lat) ** 2
            + (cluster_coords[:, 1] - centroid_lng) ** 2
        )
        radius_m = float(distances_from_centroid.max() * 111_000)

        # Count occurrences of each crime type in this hotspot cluster
        cluster_crime_types = [
            crime_types[i] for i, in_cluster in enumerate(mask) if in_cluster
        ]
        crime_type_counts: dict[str, int] = {}
        for ct in cluster_crime_types:
            crime_type_counts[ct] = crime_type_counts.get(ct, 0) + 1

        # Find the most common suburb name in this hotspot
        cluster_suburbs = [
            suburbs[i] for i, in_cluster in enumerate(mask) if in_cluster
        ]
        non_empty_suburbs = [s for s in cluster_suburbs if s]
        if non_empty_suburbs:
            suburb_counts: dict[str, int] = {}
            for s in non_empty_suburbs:
                suburb_counts[s] = suburb_counts.get(s, 0) + 1
            dominant_suburb = max(suburb_counts, key=suburb_counts.get)
        else:
            dominant_suburb = "Unknown"

        # Dominant crime type = the one that appears most frequently
        dominant_crime = (
            max(crime_type_counts, key=crime_type_counts.get)
            if crime_type_counts
            else "Unknown"
        )

        # ---- Risk level classification based on incident count --------------
        # These thresholds are designed for a single-city dataset like Harare.
        # Adjust as the data volume grows.
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
            "cluster_id":           int(label),
            # "area" is used as the dominant crime type column by HotspotView
            "area":                 dominant_crime,
            # "suburb" is used as the location/place-name column by HotspotView
            "suburb":               dominant_suburb,
            "centre_lat":           round(centroid_lat, 6),
            "centre_lng":           round(centroid_lng, 6),
            "radius_m":             round(radius_m, 1),
            "incident_count":       count,
            "risk_level":           risk_level,
            "crime_type_breakdown": crime_type_counts,
        })

    # Sort hotspots by number of crimes, largest first
    hotspots.sort(key=lambda h: h["incident_count"], reverse=True)
    return hotspots