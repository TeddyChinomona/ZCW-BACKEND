"""
serial_crime_linkage.py  —  ZIM CRIME WATCH
============================================================
Serial Crime Linkage Model

PURPOSE
-------
Determines the likelihood that two or more crimes were committed by the
same person or group (a "serial offender") by computing a composite
similarity score across five dimensions:

  1. TEMPORAL    – how close in date & time the crimes occurred
  2. SPATIAL     – how similar the incident/residential locations are
  3. MODUS OPERANDI – how similar the stolen-property descriptions are
                    (TF-IDF cosine similarity on free text)
  4. COMPLAINANT AGE    – proximity in victim age profiles
  5. COMPLAINANT GENDER – proportion of gender overlap across victims

The pairwise similarity matrix is then fed into:
  • DBSCAN clustering  – fully unsupervised; groups crimes into serial
                         clusters with no labelled data required.
  • GradientBoosting   – supervised mode (once analysts label cases as
                         linked/unlinked) for binary linkage prediction.

ARCHITECTURE  (mirrors ProfileMatcher in ml_utils.py)
------------------------------------------------------
  SerialCrimeLinkageModel.train_unsupervised(df) → cluster assignments
  SerialCrimeLinkageModel.train_supervised(df, labels) → GBT model
  SerialCrimeLinkageModel.link_probability(case_a_dict, case_b_dict)
      → float [0.0 – 1.0]  probability they share an offender
  SerialCrimeLinkageModel.cluster_cases(df) → df with 'serial_cluster'
  SerialCrimeLinkageModel.save() / .load()

DJANGO INTEGRATION
------------------
Drop this file into  zimcrimewatch/serial_linkage.py
Call from a management command:
    from zimcrimewatch.serial_linkage import SerialCrimeLinkageModel
    model = SerialCrimeLinkageModel()
    results = model.train_unsupervised_from_queryset(CrimeIncident.objects.all())

STANDALONE DEMO
---------------
Run:  python serial_crime_linkage.py
      (uses the bundled synthetic dataset or dataset1.csv)
"""

from __future__ import annotations

import logging
import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model persistence path (Django: adjust to settings.BASE_DIR / "ml_models")
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "serial_linkage_model.pkl"


# ===========================================================================
#  Feature Engineering Helpers
# ===========================================================================

def _parse_time_to_minutes(time_str) -> Optional[float]:
    """Convert messy time strings like '0905h', '1200hs', '14:30' → minutes."""
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    s = str(time_str).strip().upper().replace("HS", "").replace("H", "")
    s = s.replace(":", "")
    try:
        s = re.sub(r"[^0-9]", "", s)
        if len(s) <= 2:
            return float(s) * 60
        if len(s) == 3:
            s = "0" + s
        h, m = int(s[:2]), int(s[2:4])
        return float(h * 60 + m)
    except Exception:
        return None


def _parse_date_to_ordinal(date_str) -> Optional[float]:
    """Parse DD/MM/YY or DD/MM/YYYY strings → days since epoch."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return float(datetime.strptime(str(date_str).strip(), fmt).toordinal())
        except ValueError:
            continue
    return None


def _normalise_gender(val) -> Optional[str]:
    if pd.isna(val):
        return None
    v = str(val).strip().upper()
    if v in ("M", "MALE"):
        return "M"
    if v in ("F", "FEMALE"):
        return "F"
    return None


def _location_similarity(loc_a: str, loc_b: str) -> float:
    """
    Simple token-overlap Jaccard similarity between two address strings.
    Upgraded to TF-IDF cosine when > 100 cases are available.
    """
    if not loc_a or not loc_b:
        return 0.5  # unknown → neutral
    tokens_a = set(re.sub(r"[^a-z0-9 ]", "", loc_a.lower()).split())
    tokens_b = set(re.sub(r"[^a-z0-9 ]", "", loc_b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.5
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ===========================================================================
#  Case-Level Feature Aggregation
#  (one crime case → multiple complainants in dataset → aggregate)
# ===========================================================================

def aggregate_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple complainant rows per case_number into a single
    case-level feature row with aggregate statistics.

    Returns a DataFrame with one row per unique case.
    """
    df = df.copy()

    # Normalise gender column
    df["sex_norm"] = df["sex"].apply(_normalise_gender)
    df["date_ord"] = df["date_received"].apply(_parse_date_to_ordinal)
    df["time_min"] = df["time_received"].apply(_parse_time_to_minutes)

    agg = df.groupby("case_number").agg(
        date_ord=("date_ord", "first"),
        time_min=("time_min", "first"),
        mean_age=("age", "mean"),
        age_range=("age", lambda x: x.max() - x.min() if x.notna().sum() > 0 else 0),
        n_complainants=("complainant_name", "count"),
        pct_female=("sex_norm", lambda x: (x == "F").sum() / max(x.notna().sum(), 1)),
        pct_male=("sex_norm", lambda x: (x == "M").sum() / max(x.notna().sum(), 1)),
        location=("residential_address", lambda x: " ".join(x.dropna().astype(str).unique())),
        incident_location=("incident_location",
                           lambda x: " ".join(x.dropna().astype(str).unique())),
        mo_text=("property_stolen_description",
                 lambda x: " ".join(x.dropna().astype(str).unique())),
    ).reset_index()

    # Merge location fields for MO proxy
    agg["full_location"] = (
        agg["location"].fillna("") + " " + agg["incident_location"].fillna("")
    ).str.strip()

    return agg


# ===========================================================================
#  Pairwise Similarity Matrix
# ===========================================================================

def build_pairwise_similarity_matrix(agg_df: pd.DataFrame,
                                     tfidf: Optional[TfidfVectorizer] = None
                                     ) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Build an (N × N) weighted composite similarity matrix.

    Weights (tunable — reflect investigative importance):
      temporal   : 0.20
      spatial    : 0.25
      MO text    : 0.25
      victim age : 0.15
      victim sex : 0.15
    """
    n = len(agg_df)
    sim = np.zeros((n, n))

    # --- MO TF-IDF --------------------------------------------------------
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                                stop_words="english")
    mo_matrix = tfidf.fit_transform(agg_df["mo_text"].fillna("unknown")).toarray()
    mo_cos = cosine_similarity(mo_matrix)  # (N, N)

    # --- Temporal distance ------------------------------------------------
    scaler_time = MinMaxScaler()
    dates = agg_df["date_ord"].values.reshape(-1, 1)
    times = agg_df["time_min"].values.reshape(-1, 1)

    # Fill NaN with column mean before scaling
    date_filled = np.where(np.isnan(dates), np.nanmean(dates), dates)
    time_filled = np.where(np.isnan(times), np.nanmean(times), times)

    dates_norm = scaler_time.fit_transform(date_filled).flatten()
    scaler_t2 = MinMaxScaler()
    times_norm = scaler_t2.fit_transform(time_filled).flatten()

    # --- Numeric victim features ------------------------------------------
    ages = agg_df["mean_age"].fillna(agg_df["mean_age"].mean()).values
    pct_f = agg_df["pct_female"].fillna(0.5).values
    pct_m = agg_df["pct_male"].fillna(0.5).values
    locations = agg_df["full_location"].fillna("").values

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim[i, j] = 1.0
                continue

            # 1. Temporal similarity  (1 − normalised absolute diff)
            date_sim = 1.0 - abs(dates_norm[i] - dates_norm[j])
            time_sim = 1.0 - abs(times_norm[i] - times_norm[j])
            temporal = (date_sim * 0.6 + time_sim * 0.4)

            # 2. Spatial similarity
            spatial = _location_similarity(locations[i], locations[j])

            # 3. Modus Operandi (TF-IDF cosine)
            mo_s = float(mo_cos[i, j])

            # 4. Victim age proximity  (Gaussian decay: σ = 10 years)
            age_diff = abs(ages[i] - ages[j])
            age_s = float(np.exp(-(age_diff ** 2) / (2 * 10 ** 2)))

            # 5. Victim gender profile overlap
            gender_s = 1.0 - 0.5 * (abs(pct_f[i] - pct_f[j]) +
                                     abs(pct_m[i] - pct_m[j]))

            # Weighted composite
            composite = (
                0.20 * temporal +
                0.25 * spatial +
                0.25 * mo_s +
                0.15 * age_s +
                0.15 * gender_s
            )
            sim[i, j] = composite
            sim[j, i] = composite

    return sim, tfidf


# ===========================================================================
#  Serial Crime Linkage Model Class
# ===========================================================================

class SerialCrimeLinkageModel:
    """
    Main model class — mirrors ProfileMatcher in ml_utils.py.

    Modes
    -----
    unsupervised  : DBSCAN clusters cases by composite similarity.
                    No labels required — works from day one.

    supervised    : GradientBoostingClassifier predicts link probability
                    between a pair of cases once analysts provide labels.

    Usage (unsupervised)
    --------------------
    model = SerialCrimeLinkageModel()
    results = model.train_unsupervised(df)       # df = raw CSV or ORM data
    clustered = model.cluster_cases(df)          # returns df + 'serial_cluster'

    Usage (supervised — after labelling)
    -------------------------------------
    model = SerialCrimeLinkageModel()
    model.train_supervised(pair_df, link_labels)
    prob = model.link_probability(case_a_features, case_b_features)

    Django Management Command Integration
    --------------------------------------
    results = model.train_unsupervised_from_queryset(
        CrimeIncident.objects.all()
    )
    """

    # Similarity weights — exposed for tuning
    WEIGHTS = {
        "temporal": 0.20,
        "spatial":  0.25,
        "mo_text":  0.25,
        "age":      0.15,
        "gender":   0.15,
    }

    # DBSCAN params — eps is in similarity space (0–1), 1−eps = max distance
    DBSCAN_EPS = 0.35          # cases closer than 0.35 distance are neighbours
    DBSCAN_MIN_SAMPLES = 2     # minimum 2 crimes to form a serial cluster

    def __init__(self):
        self.tfidf: Optional[TfidfVectorizer] = None
        self.gbt: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.sim_matrix_: Optional[np.ndarray] = None
        self.agg_df_: Optional[pd.DataFrame] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self._is_supervised = False
        self._training_metrics: dict = {}

    # ------------------------------------------------------------------
    # PUBLIC API  — Unsupervised
    # ------------------------------------------------------------------

    def train_unsupervised(self, df: pd.DataFrame) -> dict:
        """
        Full unsupervised pipeline:
          1. Aggregate complainant rows → case rows
          2. Build pairwise similarity matrix
          3. DBSCAN cluster cases

        Returns dict with cluster summary.
        """
        logger.info("Aggregating %d rows into case-level features …", len(df))
        self.agg_df_ = aggregate_case_features(df)
        n = len(self.agg_df_)

        if n < 2:
            return {"error": "Need at least 2 cases to perform linkage analysis."}

        logger.info("Building %d×%d pairwise similarity matrix …", n, n)
        self.sim_matrix_, self.tfidf = build_pairwise_similarity_matrix(
            self.agg_df_, self.tfidf
        )

        # DBSCAN operates on *distance* matrix (1 − similarity)
        distance_matrix = 1.0 - self.sim_matrix_
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0, 1)

        logger.info("Running DBSCAN (eps=%.2f, min_samples=%d) …",
                    self.DBSCAN_EPS, self.DBSCAN_MIN_SAMPLES)
        dbscan = DBSCAN(
            eps=self.DBSCAN_EPS,
            min_samples=self.DBSCAN_MIN_SAMPLES,
            metric="precomputed",
        )
        self.cluster_labels_ = dbscan.fit_predict(distance_matrix)

        # Silhouette score (only meaningful if > 1 cluster found)
        n_clusters = len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0)
        sil_score = None
        if n_clusters > 1 and n > n_clusters:
            try:
                sil_score = round(float(silhouette_score(
                    distance_matrix, self.cluster_labels_, metric="precomputed"
                )), 4)
            except Exception:
                pass

        # Build summary
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

    def cluster_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the aggregated case DataFrame enriched with:
          serial_cluster  : int   (-1 = no link found, 0,1,2… = serial group)
          max_similarity  : float (highest similarity to any case in same cluster)
          linked_cases    : list  (case_number of most similar cases)
        """
        results = self.train_unsupervised(df)
        if "error" in results:
            raise ValueError(results["error"])

        out = self.agg_df_.copy()
        out["serial_cluster"] = self.cluster_labels_
        out["cluster_label"] = out["serial_cluster"].apply(
            lambda c: f"Serial Group {c}" if c >= 0 else "Unlinked"
        )

        # For each case find its most similar neighbour
        n = len(out)
        max_sims, linked = [], []
        for i in range(n):
            row_sim = self.sim_matrix_[i].copy()
            row_sim[i] = 0.0          # exclude self
            best_j = int(np.argmax(row_sim))
            max_sims.append(round(float(row_sim[best_j]), 4))
            linked.append(int(out.iloc[best_j]["case_number"]))

        out["max_similarity_score"] = max_sims
        out["most_similar_case"] = linked
        return out

    # ------------------------------------------------------------------
    # PUBLIC API  — Supervised
    # ------------------------------------------------------------------

    def train_supervised(self, pair_features: np.ndarray,
                         link_labels: np.ndarray) -> dict:
        """
        Train a GradientBoostingClassifier on labelled (case_A, case_B)
        pairs.

        pair_features : (N_pairs × 5) array — one row per pair with columns:
            [temporal_sim, spatial_sim, mo_sim, age_sim, gender_sim]
        link_labels   : (N_pairs,) int array — 1 = linked, 0 = unlinked

        Returns training metrics dict.
        """
        if len(pair_features) < 10:
            return {"error": "Need at least 10 labelled pairs to train."}

        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer

        # Impute NaNs before scaling
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(pair_features)
        self._imputer = imputer

        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X_imputed)
        y = link_labels

        # HistGBT natively handles NaNs and imbalanced data well
        self.gbt = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
        cv_scores = cross_val_score(
            self.gbt, X, y, cv=min(5, max(2, len(y) // 5)),
            scoring="roc_auc"
        )
        self.gbt.fit(X, y)
        self._is_supervised = True
        self._training_metrics["supervised"] = {
            "status": "trained_supervised",
            "n_pairs": len(y),
            "n_linked": int(y.sum()),
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        }
        self._save()
        return self._training_metrics["supervised"]

    def link_probability(self,
                         case_a: dict,
                         case_b: dict) -> dict:
        """
        Compute linkage probability between two case feature dicts.

        Each dict must contain:
            date_ord        : float  (datetime.toordinal())
            time_min        : float  (minutes since midnight, or None)
            mean_age        : float  (average victim age)
            pct_female      : float  (0.0 – 1.0)
            pct_male        : float  (0.0 – 1.0)
            full_location   : str
            mo_text         : str

        Returns:
            {
              "composite_similarity": float,
              "link_probability": float,   # GBT if available else similarity
              "feature_scores": {...},
              "verdict": str
            }
        """
        # Temporal
        d_a = case_a.get("date_ord") or 0.0
        d_b = case_b.get("date_ord") or 0.0
        t_a = case_a.get("time_min") or 720.0   # default noon
        t_b = case_b.get("time_min") or 720.0
        date_sim = 1.0 - min(abs(d_a - d_b) / max(30, 1), 1.0)  # normalise ~30 days
        time_sim = 1.0 - min(abs(t_a - t_b) / 720.0, 1.0)
        temporal = 0.6 * date_sim + 0.4 * time_sim

        # Spatial
        spatial = _location_similarity(
            case_a.get("full_location", ""),
            case_b.get("full_location", "")
        )

        # MO TF-IDF
        if self.tfidf is not None:
            vecs = self.tfidf.transform([
                case_a.get("mo_text", "") or "",
                case_b.get("mo_text", "") or "",
            ]).toarray()
            mo_s = float(cosine_similarity([vecs[0]], [vecs[1]])[0, 0])
        else:
            mo_s = 0.5

        # Age
        age_diff = abs((case_a.get("mean_age") or 35) - (case_b.get("mean_age") or 35))
        age_s = float(np.exp(-(age_diff ** 2) / (2 * 10 ** 2)))

        # Gender
        gender_s = 1.0 - 0.5 * (
            abs((case_a.get("pct_female") or 0.5) - (case_b.get("pct_female") or 0.5)) +
            abs((case_a.get("pct_male") or 0.5) - (case_b.get("pct_male") or 0.5))
        )

        feat = np.array([[temporal, spatial, mo_s, age_s, gender_s]])
        feat = np.nan_to_num(feat, nan=0.5)
        composite = float(np.dot(
            feat[0],
            [self.WEIGHTS["temporal"], self.WEIGHTS["spatial"],
             self.WEIGHTS["mo_text"], self.WEIGHTS["age"], self.WEIGHTS["gender"]]
        ))

        # Use GBT if trained, else use composite similarity as probability
        if self._is_supervised and self.gbt is not None and self.scaler is not None:
            feat_imputed = self._imputer.transform(feat)
            feat_scaled = self.scaler.transform(feat_imputed)
            prob = float(self.gbt.predict_proba(feat_scaled)[0, 1])
        else:
            prob = composite

        verdict = (
            "HIGH – Likely same offender" if prob >= 0.70 else
            "MODERATE – Possible link, investigate further" if prob >= 0.45 else
            "LOW – Unlikely to be linked"
        )

        return {
            "composite_similarity": round(composite, 4),
            "link_probability": round(prob, 4),
            "feature_scores": {
                "temporal_similarity": round(temporal, 4),
                "spatial_similarity": round(spatial, 4),
                "mo_similarity": round(mo_s, 4),
                "age_similarity": round(age_s, 4),
                "gender_similarity": round(gender_s, 4),
            },
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Django Integration Helper
    # ------------------------------------------------------------------

    def train_unsupervised_from_queryset(self, incidents_qs) -> dict:
        """
        Django-ready wrapper. Pass in a CrimeIncident QuerySet.
        The queryset must expose at minimum:
            case_number, date_received, time_received, complainant_name,
            sex, age, residential_address, incident_location,
            property_stolen_description (or modus_operandi)
        """
        data = list(incidents_qs.values(
            "case_number", "date_received", "time_received",
            "complainant_name", "sex", "age",
            "residential_address", "incident_location",
            "property_stolen_description",
        ))
        if not data:
            return {"error": "Empty queryset."}
        df = pd.DataFrame(data)
        return self.train_unsupervised(df)

    # ------------------------------------------------------------------
    # Cluster Summary Helper
    # ------------------------------------------------------------------

    def _build_cluster_summary(self) -> list[dict]:
        if self.cluster_labels_ is None or self.agg_df_ is None:
            return []
        summaries = []
        unique_clusters = sorted(set(self.cluster_labels_))
        for c in unique_clusters:
            if c == -1:
                continue
            mask = self.cluster_labels_ == c
            cases = self.agg_df_[mask]["case_number"].tolist()
            sim_vals = []
            idxs = np.where(mask)[0]
            for i in idxs:
                for j in idxs:
                    if i < j:
                        sim_vals.append(self.sim_matrix_[i, j])

            summaries.append({
                "cluster_id": int(c),
                "label": f"Serial Group {c}",
                "n_cases": int(mask.sum()),
                "case_numbers": cases,
                "mean_intra_similarity": round(float(np.mean(sim_vals)), 4) if sim_vals else 1.0,
                "min_intra_similarity": round(float(np.min(sim_vals)), 4) if sim_vals else 1.0,
            })
        return sorted(summaries, key=lambda x: x["n_cases"], reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info("SerialCrimeLinkageModel saved → %s", MODEL_PATH)

    @classmethod
    def load(cls) -> "SerialCrimeLinkageModel":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Serial linkage model not found. Run train_unsupervised() first."
            )
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_and_cluster(cls, df: pd.DataFrame) -> pd.DataFrame:
        try:
            instance = cls.load()
        except FileNotFoundError:
            instance = cls()
        return instance.cluster_cases(df)

    @classmethod
    def load_and_link(cls, case_a: dict, case_b: dict) -> dict:
        instance = cls.load()
        return instance.link_probability(case_a, case_b)


# ===========================================================================
#  Standalone Demo / Self-Test
# ===========================================================================

if __name__ == "__main__":
    import os

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    csv_path = Path(__file__).parent / "dataset1.csv"
    if csv_path.exists():
        df_raw = pd.read_csv(csv_path, on_bad_lines="skip")
        print(f"✓ Loaded dataset1.csv  ({len(df_raw)} rows, "
              f"{df_raw['case_number'].nunique()} unique cases)\n")
    else:
        # Synthetic fallback dataset
        print("dataset1.csv not found — using synthetic data\n")
        df_raw = pd.DataFrame({
            "case_number":   [1,1,2,2,3,4,4,5,6,7,8,9,10,10],
            "date_received": ["01/01/24"]*4 + ["03/01/24"]*4 + ["05/01/24"]*6,
            "time_received": ["0900h","0900h","0910h","0910h","1400h","1405h",
                              "1405h","0855h","0856h","0857h","0858h","0859h","1200hs","1200hs"],
            "complainant_name": ["Alice","Bob","Eve","Frank","Carol","Dave","Gina",
                                 "Unknown","Unknown","Unknown","Unknown","Unknown","Han","Ivy"],
            "sex":    ["F","M","F","M","F","M","F","M","F","M","F","M","F","M"],
            "age":    [25,45,27,43,30,28,32,60,22,55,21,44,29,47],
            "residential_address": ["Highlands","Highlands","Highlands",
                                    "Highlands","Belvedere","Belvedere",
                                    "Belvedere","Mbare","Mbare","Mbare",
                                    "Avondale","Avondale","CBD","CBD"],
            "incident_location": ["College Bar","College Bar","College Bar",
                                   "College Bar","Shop A","Shop A","Shop A",
                                   None,None,None,None,None,"Market","Market"],
            "property_stolen_description": [
                "Cellphone Tablet","Cellphone Tablet","Cellphone","Tablet",
                "Burglar Bar","Burglar Bar","Burglar Bar broken window",
                None,None,None,None,None,"Cash","Cash wallet"],
        })

    # -----------------------------------------------------------------------
    # 1. Unsupervised clustering
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(" STAGE 1: Unsupervised Serial Crime Clustering")
    print("=" * 60)
    model = SerialCrimeLinkageModel()
    results = model.train_unsupervised(df_raw)

    print(f"\n  Cases analysed      : {results['n_cases']}")
    print(f"  Serial clusters found: {results['n_serial_clusters']}")
    print(f"  Unlinked (noise)     : {results['n_unlinked_cases']}")
    if results.get("silhouette_score") is not None:
        print(f"  Silhouette score     : {results['silhouette_score']}")

    if results["clusters"]:
        print("\n  Cluster Details:")
        for c in results["clusters"]:
            print(f"    [{c['label']}]  {c['n_cases']} cases → "
                  f"cases {c['case_numbers']}  "
                  f"(avg similarity: {c['mean_intra_similarity']})")

    # -----------------------------------------------------------------------
    # 2. Full clustered DataFrame
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STAGE 2: Case-Level Cluster Assignments")
    print("=" * 60)
    clustered_df = model.cluster_cases(df_raw)
    display_cols = ["case_number", "cluster_label", "max_similarity_score",
                    "most_similar_case", "mean_age", "pct_female", "date_ord"]
    print(clustered_df[[c for c in display_cols if c in clustered_df.columns]].to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. Pairwise linkage probability
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STAGE 3: Pairwise Linkage Assessment")
    print("=" * 60)

    agg = model.agg_df_
    if len(agg) >= 2:
        for (i, j) in [(0, 1), (0, len(agg) - 1)]:
            row_a = agg.iloc[i]
            row_b = agg.iloc[j]
            case_a_dict = {
                "date_ord": row_a["date_ord"],
                "time_min": row_a["time_min"],
                "mean_age": row_a["mean_age"],
                "pct_female": row_a["pct_female"],
                "pct_male": row_a["pct_male"],
                "full_location": row_a["full_location"],
                "mo_text": row_a["mo_text"],
            }
            case_b_dict = {
                "date_ord": row_b["date_ord"],
                "time_min": row_b["time_min"],
                "mean_age": row_b["mean_age"],
                "pct_female": row_b["pct_female"],
                "pct_male": row_b["pct_male"],
                "full_location": row_b["full_location"],
                "mo_text": row_b["mo_text"],
            }
            result = model.link_probability(case_a_dict, case_b_dict)
            print(f"\n  Case {int(row_a['case_number'])} ←→ Case {int(row_b['case_number'])}")
            print(f"    Link Probability : {result['link_probability']:.1%}")
            print(f"    Verdict          : {result['verdict']}")
            print(f"    Feature Scores   :")
            for feat, score in result["feature_scores"].items():
                score_val = score if not (isinstance(score, float) and np.isnan(score)) else 0.0
                bar = "█" * int(score_val * 20)
                print(f"      {feat:<25} {score_val:.3f}  {bar}")

    # -----------------------------------------------------------------------
    # 4. Synthetic supervised demo (auto-generated labels)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STAGE 4: Supervised Training Demo (synthetic labels)")
    print("=" * 60)
    n = len(agg)
    if n >= 4:
        pairs_X = []
        pairs_y = []
        for i in range(n):
            for j in range(i + 1, n):
                composite = float(model.sim_matrix_[i, j])
                row_a = agg.iloc[i]
                row_b = agg.iloc[j]
                # Temporal
                d_a = row_a["date_ord"] or 0.0
                d_b = row_b["date_ord"] or 0.0
                t_a = row_a["time_min"] or 720.0
                t_b = row_b["time_min"] or 720.0
                date_sim = 1.0 - min(abs(d_a - d_b) / 30.0, 1.0)
                time_sim = 1.0 - min(abs(t_a - t_b) / 720.0, 1.0)
                temporal = 0.6 * date_sim + 0.4 * time_sim
                spatial = _location_similarity(row_a["full_location"], row_b["full_location"])
                if model.tfidf:
                    vecs = model.tfidf.transform([row_a["mo_text"] or "", row_b["mo_text"] or ""]).toarray()
                    mo_s = float(cosine_similarity([vecs[0]], [vecs[1]])[0, 0])
                else:
                    mo_s = 0.5
                age_diff = abs((row_a["mean_age"] or 35) - (row_b["mean_age"] or 35))
                age_s = float(np.exp(-(age_diff ** 2) / 200))
                gender_s = 1.0 - 0.5 * (abs(row_a["pct_female"] - row_b["pct_female"]) +
                                         abs(row_a["pct_male"] - row_b["pct_male"]))
                pairs_X.append([temporal, spatial, mo_s, age_s, gender_s])
                # Auto-label: linked if same cluster and cluster != -1
                ca = model.cluster_labels_[i]
                cb = model.cluster_labels_[j]
                pairs_y.append(1 if (ca == cb and ca != -1) else 0)

        X_arr = np.array(pairs_X)
        y_arr = np.array(pairs_y)

        print(f"  Generated {len(y_arr)} pairs  "
              f"({y_arr.sum()} linked, {(~y_arr.astype(bool)).sum()} unlinked)")

        if y_arr.sum() >= 2 and (~y_arr.astype(bool)).sum() >= 2:
            sup_results = model.train_supervised(X_arr, y_arr)
            if "error" not in sup_results:
                print(f"  CV ROC-AUC : {sup_results['cv_roc_auc_mean']:.4f} "
                      f"(±{sup_results['cv_roc_auc_std']:.4f})")
                print("  Weights used: temporal=0.20, spatial=0.25, mo=0.25, age=0.15, gender=0.15")
            else:
                print(f"  Supervised skipped: {sup_results['error']}")
        else:
            print("  Supervised training skipped: need both linked & unlinked pairs")

    print("\n✓ Serial Crime Linkage Model — self-test complete.")
    print(f"  Model saved → {MODEL_PATH}")
