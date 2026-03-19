"""
serial_crime_linkage.py  —  ZimCrimeWatch Serial Crime Linkage Model
=====================================================================

PURPOSE
-------
Figures out whether multiple crimes were committed by the same person
(a "serial offender") by scoring how similar pairs of crimes are across
five dimensions:

  1. TEMPORAL    — how close together the crimes happened in date & time
  2. SPATIAL     — how similar the crime locations are (Jaccard overlap)
  3. MODUS OPERANDI (MO) — how similar the stolen-property descriptions are
                            (TF-IDF cosine similarity on free text)
  4. VICTIM AGE  — how close the victims' ages are (Gaussian similarity)
  5. VICTIM GENDER — how similar the gender profile of victims is

Those five scores are combined into one composite similarity score per pair
of cases, building an (N × N) similarity matrix across all N cases.

That matrix feeds into:
  • DBSCAN clustering  — groups crimes into "serial clusters" with no labels needed
  • GradientBoosting   — once analysts label pairs as linked/unlinked, this
                          predicts the probability that two new cases are linked

HOW THE DATA IS STRUCTURED
---------------------------
The raw data has one row per complainant (victim), but multiple complainants
can belong to the same criminal case. The case_number column ties them together.

We first "aggregate" — collapsing all complainant rows for a case into a
single case-level summary row — before computing similarities.

DJANGO USAGE
------------
    from zimcrimewatch.serial_crime_linkage import SerialCrimeLinkageModel
    model = SerialCrimeLinkageModel()
    results = model.train_unsupervised_from_queryset(CrimeIncident.objects.all())
"""

import logging
import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Path where the trained model is saved on disk (next to this file)
MODEL_PATH = Path(__file__).parent / "serial_linkage_model.pkl"


# =============================================================================
# Helper functions  —  small, focused utilities used by the main functions
# =============================================================================

def _parse_time_to_minutes(time_str):
    """
    Convert a messy time string into minutes since midnight (a plain float).

    Handles formats like: '0905h', '0905hs', '9:05', '14:30', '900'
    Returns None if the string cannot be parsed.

    Example:
        _parse_time_to_minutes("0905h")  →  545.0   (9*60 + 5)
        _parse_time_to_minutes("14:30")  →  870.0   (14*60 + 30)
    """
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None

    # Strip trailing noise like "h", "hs", spaces; remove colons for uniform format
    s = str(time_str).strip().upper().replace("HS", "").replace("H", "").replace(":", "")

    try:
        # Remove any remaining non-digit characters
        s = re.sub(r"[^0-9]", "", s)

        # Short strings like "9" or "14" are just hours
        if len(s) <= 2:
            return float(s) * 60

        # Pad 3-digit strings like "905" → "0905"
        if len(s) == 3:
            s = "0" + s

        # Now s is 4+ digits: first 2 are hours, next 2 are minutes
        hours = int(s[:2])
        minutes = int(s[2:4])
        return float(hours * 60 + minutes)

    except Exception:
        return None


def _parse_date_to_ordinal(date_str):
    """
    Convert a date string into an ordinal number (days since year 0001).

    Using ordinal numbers lets us subtract dates to get "days apart".

    Handles: 'DD/MM/YY', 'DD/MM/YYYY', 'YYYY-MM-DD'
    Returns None if the string cannot be parsed.
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None

    # Try each format until one works
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return float(datetime.strptime(str(date_str).strip(), fmt).toordinal())
        except ValueError:
            continue  # try next format

    return None  # no format worked


def _normalise_gender(value):
    """
    Standardise gender strings to 'M', 'F', or None.
    Handles: 'male', 'MALE', 'M', 'female', 'FEMALE', 'F'
    """
    if pd.isna(value):
        return None
    v = str(value).strip().upper()
    if v in ("M", "MALE"):
        return "M"
    if v in ("F", "FEMALE"):
        return "F"
    return None  # unknown / other


def _location_similarity(loc_a, loc_b):
    """
    Measure how similar two address strings are using Jaccard similarity.

    Jaccard similarity = (words in common) / (total unique words in both).
    Example:
        "Highlands Harare" vs "Highlands CBD"
        intersection = {"highlands"}, union = {"highlands", "harare", "cbd"}
        similarity = 1/3 ≈ 0.33

    Returns 0.5 (neutral) if either location is missing.
    Returns values between 0 (no overlap) and 1 (identical).
    """
    if not loc_a or not loc_b:
        return 0.5  # unknown location → neutral score, not zero

    # Lowercase and keep only letters, digits, and spaces
    clean_a = re.sub(r"[^a-z0-9 ]", "", loc_a.lower())
    clean_b = re.sub(r"[^a-z0-9 ]", "", loc_b.lower())

    tokens_a = set(clean_a.split())
    tokens_b = set(clean_b.split())

    if not tokens_a or not tokens_b:
        return 0.5

    intersection = tokens_a & tokens_b   # words in both
    union = tokens_a | tokens_b          # all unique words

    return len(intersection) / len(union)


# =============================================================================
# Case-Level Feature Aggregation
# "Collapse multiple victim rows per case into one summary row"
# =============================================================================

def aggregate_case_features(df):
    """
    The raw data can have multiple complainant rows per case_number.
    This function collapses them into one row per case with summary statistics.

    Input columns needed:
        case_number, date_received, time_received, complainant_name,
        sex, age, residential_address, incident_location,
        property_stolen_description

    Output: one row per unique case_number with aggregated features.
    """
    df = df.copy()

    # Convert raw columns to numeric formats we can do maths on
    df["sex_norm"] = df["sex"].apply(_normalise_gender)
    df["date_ord"] = df["date_received"].apply(_parse_date_to_ordinal)
    df["time_min"] = df["time_received"].apply(_parse_time_to_minutes)

    # Group all complainant rows by case_number and compute one summary per case
    agg = df.groupby("case_number").agg(
        # Use the date/time of the first recorded complainant as the case date/time
        date_ord=("date_ord", "first"),
        time_min=("time_min", "first"),

        # Average and range of victim ages across the case
        mean_age=("age", "mean"),
        age_range=("age", lambda x: x.max() - x.min() if x.notna().sum() > 0 else 0),

        # Count how many complainants there were
        n_complainants=("complainant_name", "count"),

        # Proportion of female and male victims (values between 0 and 1)
        pct_female=("sex_norm", lambda x: (x == "F").sum() / max(x.notna().sum(), 1)),
        pct_male=("sex_norm", lambda x: (x == "M").sum() / max(x.notna().sum(), 1)),

        # Combine all unique location strings into one text blob for comparison
        location=("residential_address",
                  lambda x: " ".join(x.dropna().astype(str).unique())),
        incident_location=("incident_location",
                           lambda x: " ".join(x.dropna().astype(str).unique())),

        # Combine all stolen property descriptions into one text blob
        mo_text=("property_stolen_description",
                 lambda x: " ".join(x.dropna().astype(str).unique())),
    ).reset_index()

    # Merge the two location fields into one combined text for similarity matching
    agg["full_location"] = (
        agg["location"].fillna("") + " " + agg["incident_location"].fillna("")
    ).str.strip()

    return agg


# =============================================================================
# Pairwise Similarity Matrix
# "Score every pair of cases across all five similarity dimensions"
# =============================================================================

def build_pairwise_similarity_matrix(agg_df, tfidf=None):
    """
    Build an (N × N) matrix where entry [i, j] is the composite similarity
    score between case i and case j.  The matrix is symmetric (sim[i,j] == sim[j,i])
    and has 1.0 on the diagonal (a case is perfectly similar to itself).

    The five component scores are weighted and summed:
      Temporal   × 0.20
      Spatial    × 0.25
      MO text    × 0.25
      Victim age × 0.15
      Gender     × 0.15
      ─────────────────
      Total       1.00

    Parameters:
        agg_df : DataFrame produced by aggregate_case_features()
        tfidf  : an already-fitted TfidfVectorizer (pass None to fit a new one)

    Returns:
        (similarity_matrix, tfidf_vectorizer)
    """
    n = len(agg_df)

    # ---- Fit TF-IDF on modus operandi text if not already fitted -----------
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))

    # Encode all MO texts into TF-IDF vectors (one row per case)
    mo_texts = agg_df["mo_text"].fillna("").tolist()
    mo_matrix = tfidf.fit_transform(mo_texts)

    # Compute cosine similarity between all pairs of MO text vectors at once.
    # Result is an (N × N) matrix where mo_cos[i, j] is the text similarity.
    mo_cos = cosine_similarity(mo_matrix)

    # ---- Extract numeric columns for the other similarity dimensions --------
    locations = agg_df["full_location"].fillna("").tolist()

    # Fill missing numeric values with reasonable defaults
    ages = agg_df["mean_age"].fillna(agg_df["mean_age"].median()).values
    pct_f = agg_df["pct_female"].fillna(0.5).values
    pct_m = agg_df["pct_male"].fillna(0.5).values

    # Normalise date and time to [0, 1] range using MinMaxScaler,
    # so that temporal differences are on a comparable scale to other features.
    scaler = MinMaxScaler()
    dates_raw = agg_df["date_ord"].fillna(agg_df["date_ord"].median()).values.reshape(-1, 1)
    times_raw = agg_df["time_min"].fillna(720.0).values.reshape(-1, 1)  # default noon
    dates_norm = scaler.fit_transform(dates_raw).flatten()
    times_norm = scaler.fit_transform(times_raw).flatten()

    # ---- Compute the composite similarity for every pair (i, j) ------------
    # We use an (N × N) matrix initialised to zeros, then fill it.
    sim = np.zeros((n, n), dtype=float)
    np.fill_diagonal(sim, 1.0)  # each case is 100% similar to itself

    for i in range(n):
        for j in range(i + 1, n):  # only upper triangle; we'll mirror it

            # 1. Temporal similarity: 1 minus normalised date difference,
            #    blended 60/40 with time-of-day similarity.
            date_diff = abs(dates_norm[i] - dates_norm[j])
            time_diff = abs(times_norm[i] - times_norm[j])
            temporal = 0.6 * (1.0 - date_diff) + 0.4 * (1.0 - time_diff)

            # 2. Spatial similarity: Jaccard token overlap of location strings
            spatial = _location_similarity(locations[i], locations[j])

            # 3. MO text similarity: already computed above in the full matrix
            mo_s = float(mo_cos[i, j])

            # 4. Victim age similarity: Gaussian function of age difference.
            #    np.exp(-(diff²) / (2 * σ²)) with σ=10 years:
            #    - Same age      → similarity 1.0
            #    - 10 years apart → similarity ~0.61
            #    - 20 years apart → similarity ~0.14
            age_diff = abs(ages[i] - ages[j])
            age_s = float(np.exp(-(age_diff ** 2) / (2 * 10 ** 2)))

            # 5. Gender profile similarity: 1 minus the average absolute
            #    difference in female% and male% between the two cases.
            #    If both cases have 100% female victims, score = 1.0.
            gender_s = 1.0 - 0.5 * (
                abs(pct_f[i] - pct_f[j]) +
                abs(pct_m[i] - pct_m[j])
            )

            # Weighted composite score
            composite = (
                0.20 * temporal +
                0.25 * spatial +
                0.25 * mo_s +
                0.15 * age_s +
                0.15 * gender_s
            )

            # Fill both sides of the symmetric matrix
            sim[i, j] = composite
            sim[j, i] = composite

    return sim, tfidf


# =============================================================================
# Main Model Class
# =============================================================================

class SerialCrimeLinkageModel:
    """
    The main class that ties everything together.

    Two modes:
      unsupervised — DBSCAN clusters crimes by similarity, no labels needed.
      supervised   — GradientBoosting predicts whether two cases are linked,
                     once analysts have labelled some pairs as linked/unlinked.

    Quick usage:
        model = SerialCrimeLinkageModel()
        results = model.train_unsupervised(df)
        clustered_df = model.cluster_cases(df)
    """

    # ---- Configuration: weights and DBSCAN tuning parameters ---------------
    WEIGHTS = {
        "temporal": 0.20,
        "spatial":  0.25,
        "mo_text":  0.25,
        "age":      0.15,
        "gender":   0.15,
    }

    # DBSCAN eps is the maximum *distance* (= 1 - similarity) allowed between
    # two neighbouring points. eps=0.35 means cases must have similarity >= 0.65
    # to be considered neighbours.
    DBSCAN_EPS = 0.35

    # Minimum number of cases required to form a serial cluster
    DBSCAN_MIN_SAMPLES = 2

    def __init__(self):
        self.tfidf = None              # TfidfVectorizer fitted during training
        self.gbt = None                # GradientBoostingClassifier (supervised mode)
        self.scaler = None             # MinMaxScaler for supervised feature scaling
        self.sim_matrix_ = None        # The (N × N) similarity matrix
        self.agg_df_ = None            # Case-level aggregated features
        self.cluster_labels_ = None    # DBSCAN cluster assignment per case (-1 = noise)
        self._is_supervised = False    # True once train_supervised() has been called
        self._training_metrics = {}    # Summary stats from the last training run

    # =========================================================================
    # PUBLIC API — Unsupervised Mode
    # =========================================================================

    def train_unsupervised(self, df):
        """
        Full unsupervised pipeline on a raw DataFrame:
          Step 1: Aggregate multiple complainant rows → one row per case
          Step 2: Build the (N × N) pairwise similarity matrix
          Step 3: Run DBSCAN on (1 - similarity) as a distance matrix

        Parameters:
            df : DataFrame with columns: case_number, date_received,
                 time_received, complainant_name, sex, age,
                 residential_address, incident_location,
                 property_stolen_description

        Returns:
            dict with summary statistics and cluster details
        """
        logger.info("Aggregating %d complainant rows into case-level features...", len(df))
        self.agg_df_ = aggregate_case_features(df)
        n = len(self.agg_df_)

        if n < 2:
            return {"error": "Need at least 2 cases to perform linkage analysis."}

        logger.info("Building %d × %d pairwise similarity matrix...", n, n)
        self.sim_matrix_, self.tfidf = build_pairwise_similarity_matrix(
            self.agg_df_, self.tfidf
        )

        # ---- Convert similarity to distance for DBSCAN ----------------------
        # DBSCAN needs a *distance* matrix (small = close), but we have a
        # *similarity* matrix (large = close).  Distance = 1 - similarity.
        distance_matrix = 1.0 - self.sim_matrix_

        # Set the diagonal to exactly 0.0 (a case has zero distance to itself)
        np.fill_diagonal(distance_matrix, 0.0)

        # Clip to [0, 1] to remove any tiny floating-point negatives
        distance_matrix = np.clip(distance_matrix, 0, 1)

        logger.info(
            "Running DBSCAN (eps=%.2f, min_samples=%d)...",
            self.DBSCAN_EPS, self.DBSCAN_MIN_SAMPLES
        )

        # metric="precomputed" tells DBSCAN we are providing a distance matrix
        # rather than raw data points
        dbscan = DBSCAN(
            eps=self.DBSCAN_EPS,
            min_samples=self.DBSCAN_MIN_SAMPLES,
            metric="precomputed",
        )
        # fit_predict returns an array of cluster labels, one per case.
        # -1 means the case did not fit into any cluster ("noise").
        self.cluster_labels_ = dbscan.fit_predict(distance_matrix)

        # ---- Count clusters and compute quality score -----------------------
        # The set of unique labels minus -1 gives the number of real clusters
        unique_labels = set(self.cluster_labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Silhouette score: measures how well-separated the clusters are.
        # Ranges from -1 (bad) to 1 (great). Only meaningful if > 1 cluster found.
        sil_score = None
        if n_clusters > 1 and n > n_clusters:
            try:
                sil_score = round(float(silhouette_score(
                    distance_matrix,
                    self.cluster_labels_,
                    metric="precomputed",
                )), 4)
            except Exception:
                pass  # silhouette can fail in edge cases; that's fine

        cluster_summary = self._build_cluster_summary()

        self._training_metrics = {
            "status": "trained_unsupervised",
            "n_cases": n,
            "n_serial_clusters": n_clusters,
            "n_unlinked_cases": int((self.cluster_labels_ == -1).sum()),
            "silhouette_score": sil_score,
            "clusters": cluster_summary,
        }

        self._save()
        logger.info("Model saved. %d serial cluster(s) found.", n_clusters)
        return self._training_metrics

    def cluster_cases(self, df):
        """
        Add a 'serial_cluster' column to the DataFrame using the trained model.
        Cases with cluster label -1 are noise (no serial link found).

        Parameters:
            df : raw DataFrame (same format as train_unsupervised)

        Returns:
            df with an added 'serial_cluster' column
        """
        # If there are no clusters stored yet, retrain first
        if self.cluster_labels_ is None or self.agg_df_ is None:
            self.train_unsupervised(df)

        # Map case_number → cluster_label using the stored agg_df_
        cluster_map = dict(
            zip(self.agg_df_["case_number"], self.cluster_labels_)
        )

        # Aggregate the input df so we have one row per case, then map labels
        agg = aggregate_case_features(df)
        agg["serial_cluster"] = agg["case_number"].map(cluster_map).fillna(-1).astype(int)
        return agg

    def train_unsupervised_from_queryset(self, incidents_qs):
        """
        Convenience method: pull data from a Django QuerySet and train.

        The queryset must expose these fields:
            case_number, date_received, time_received, complainant_name,
            sex, age, residential_address, incident_location,
            property_stolen_description
        """
        data = list(incidents_qs.values(
            "case_number", "date_received", "time_received",
            "complainant_name", "sex", "age",
            "residential_address", "incident_location",
            "property_stolen_description",
        ))

        if not data:
            return {"error": "No data returned from queryset."}

        df = pd.DataFrame(data)
        return self.train_unsupervised(df)

    # =========================================================================
    # PUBLIC API — Supervised Mode
    # =========================================================================

    def train_supervised(self, pair_df, link_labels):
        """
        Train a GradientBoosting classifier to predict whether two cases
        are linked to the same offender, using analyst-provided labels.

        Parameters:
            pair_df : DataFrame where each row has features for a PAIR of cases.
                      Must have columns: temporal, spatial, mo_text, age, gender
                      (the five component similarity scores for that pair).
            link_labels : array-like of 1 (linked) or 0 (not linked) per row.

        Returns:
            dict with cross-validation accuracy metrics
        """
        feature_cols = ["temporal", "spatial", "mo_text", "age", "gender"]
        X = pair_df[feature_cols].fillna(0.5).values

        # Scale features to [0, 1] — GradientBoosting benefits from scaled input
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.gbt = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

        # Cross-validate to get honest accuracy estimate
        cv_scores = cross_val_score(self.gbt, X_scaled, link_labels, cv=5, scoring="roc_auc")
        self.gbt.fit(X_scaled, link_labels)
        self._is_supervised = True

        self._save()
        logger.info(
            "Supervised model trained. AUC: %.3f ± %.3f",
            cv_scores.mean(), cv_scores.std()
        )
        return {
            "status": "trained_supervised",
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std": round(float(cv_scores.std()), 4),
            "n_pairs": len(pair_df),
        }

    def link_probability(self, case_a, case_b):
        """
        Compute how likely two cases share an offender.

        Each case dict must have these keys:
            date_ord      : float  (from datetime.toordinal())
            time_min      : float  (minutes since midnight, or None)
            mean_age      : float  (average victim age)
            pct_female    : float  (0.0 to 1.0)
            pct_male      : float  (0.0 to 1.0)
            full_location : str    (combined address text)
            mo_text       : str    (stolen property description)

        Returns:
            {
              "composite_similarity": float,
              "link_probability":     float,   # GBT score if trained, else composite
              "feature_scores":       dict,    # the five individual scores
              "verdict":              str      # human-readable conclusion
            }
        """
        # ---- Compute the five individual similarity scores ------------------

        # 1. Temporal: date similarity (60%) + time-of-day similarity (40%)
        date_a = case_a.get("date_ord") or 0.0
        date_b = case_b.get("date_ord") or 0.0
        time_a = case_a.get("time_min") or 720.0  # default noon if missing
        time_b = case_b.get("time_min") or 720.0

        # Normalise date difference: 30 days apart → similarity ~0, same day → 1.0
        date_sim = 1.0 - min(abs(date_a - date_b) / 30.0, 1.0)
        # Normalise time difference: 720 minutes = 12 hours → similarity 0, same time → 1.0
        time_sim = 1.0 - min(abs(time_a - time_b) / 720.0, 1.0)
        temporal = 0.6 * date_sim + 0.4 * time_sim

        # 2. Spatial: Jaccard token overlap of combined location strings
        spatial = _location_similarity(
            case_a.get("full_location", ""),
            case_b.get("full_location", ""),
        )

        # 3. MO text: TF-IDF cosine similarity on stolen property descriptions
        if self.tfidf is not None:
            # Transform both texts using the already-fitted TF-IDF vocabulary
            vecs = self.tfidf.transform([
                case_a.get("mo_text", "") or "",
                case_b.get("mo_text", "") or "",
            ]).toarray()
            # cosine_similarity returns (1×1) shaped result; [0,0] extracts the float
            mo_s = float(cosine_similarity([vecs[0]], [vecs[1]])[0, 0])
        else:
            mo_s = 0.5  # neutral if no TF-IDF model available

        # 4. Victim age: Gaussian decay with σ=10 years
        age_a = case_a.get("mean_age") or 35
        age_b = case_b.get("mean_age") or 35
        age_diff = abs(age_a - age_b)
        age_s = float(np.exp(-(age_diff ** 2) / (2 * 10 ** 2)))

        # 5. Gender profile: 1 minus mean absolute difference in gender %
        gender_s = 1.0 - 0.5 * (
            abs((case_a.get("pct_female") or 0.5) - (case_b.get("pct_female") or 0.5)) +
            abs((case_a.get("pct_male") or 0.5) - (case_b.get("pct_male") or 0.5))
        )

        # ---- Weighted composite score ---------------------------------------
        feature_vec = np.array([[temporal, spatial, mo_s, age_s, gender_s]])
        feature_vec = np.nan_to_num(feature_vec, nan=0.5)  # replace any NaN with 0.5

        weights = [
            self.WEIGHTS["temporal"],
            self.WEIGHTS["spatial"],
            self.WEIGHTS["mo_text"],
            self.WEIGHTS["age"],
            self.WEIGHTS["gender"],
        ]
        composite = float(np.dot(feature_vec[0], weights))

        # ---- Use GBT if trained, else use composite as the probability ------
        if self._is_supervised and self.gbt is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(feature_vec)
            # predict_proba returns [[prob_class_0, prob_class_1]]
            # index [0, 1] gives the probability that label = 1 (linked)
            link_prob = float(self.gbt.predict_proba(X_scaled)[0, 1])
        else:
            link_prob = composite  # use composite as fallback probability

        # ---- Human-readable verdict ----------------------------------------
        if link_prob >= 0.75:
            verdict = "High likelihood of serial linkage"
        elif link_prob >= 0.50:
            verdict = "Moderate likelihood — warrants investigation"
        elif link_prob >= 0.30:
            verdict = "Low likelihood — possible coincidence"
        else:
            verdict = "Unlikely to be linked"

        return {
            "composite_similarity": round(composite, 4),
            "link_probability": round(link_prob, 4),
            "feature_scores": {
                "temporal": round(temporal, 4),
                "spatial":  round(spatial, 4),
                "mo_text":  round(mo_s, 4),
                "age":      round(age_s, 4),
                "gender":   round(gender_s, 4),
            },
            "verdict": verdict,
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _build_cluster_summary(self):
        """
        Build a human-readable summary for each DBSCAN cluster found,
        including which cases are in it and how similar they are to each other.

        Returns a list of dicts, sorted by cluster size (largest first).
        """
        if self.cluster_labels_ is None or self.agg_df_ is None:
            return []

        summaries = []
        unique_clusters = sorted(set(self.cluster_labels_))

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue  # skip noise points

            # Boolean mask: True for cases assigned to this cluster
            in_cluster = self.cluster_labels_ == cluster_id
            cluster_cases = self.agg_df_[in_cluster]["case_number"].tolist()

            # Compute intra-cluster similarities (how similar are cases within the cluster?)
            cluster_indices = np.where(in_cluster)[0]
            intra_sims = []
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:  # only upper triangle to avoid counting each pair twice
                        intra_sims.append(self.sim_matrix_[i, j])

            if intra_sims:
                mean_sim = round(float(np.mean(intra_sims)), 4)
                min_sim  = round(float(np.min(intra_sims)), 4)
            else:
                mean_sim = 1.0  # only one case in cluster
                min_sim  = 1.0

            summaries.append({
                "cluster_id": int(cluster_id),
                "label": f"Serial Group {cluster_id}",
                "n_cases": int(in_cluster.sum()),
                "case_numbers": cluster_cases,
                "mean_intra_similarity": mean_sim,
                "min_intra_similarity": min_sim,
            })

        # Return largest clusters first
        return sorted(summaries, key=lambda x: x["n_cases"], reverse=True)

    # =========================================================================
    # Persistence  —  save and load the trained model
    # =========================================================================

    def _save(self):
        """
        Save the entire model object to disk using pickle.
        The saved file contains the trained DBSCAN results, TF-IDF vocabulary,
        and GBT classifier (if trained) so nothing needs to be recomputed later.
        """
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("SerialCrimeLinkageModel saved to %s", MODEL_PATH)

    @classmethod
    def load(cls):
        """
        Load a previously saved model from disk.
        Raises FileNotFoundError if train_unsupervised() has never been run.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Serial linkage model not found. Run train_unsupervised() first."
            )
        with open(MODEL_PATH, "rb") as f:
            instance = pickle.load(f)
        logger.info("SerialCrimeLinkageModel loaded from %s", MODEL_PATH)
        return instance

    @classmethod
    def load_and_cluster(cls, df):
        """
        Convenience method: load a saved model and cluster a new DataFrame.
        If no saved model exists, trains a new one on the provided data.
        """
        try:
            instance = cls.load()
        except FileNotFoundError:
            instance = cls()  # create fresh model and train below
        return instance.cluster_cases(df)

    @classmethod
    def load_and_link(cls, case_a, case_b):
        """
        Convenience method: load a saved model and compute link probability
        between two case dicts.
        """
        instance = cls.load()
        return instance.link_probability(case_a, case_b)