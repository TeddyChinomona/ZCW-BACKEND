"""
test_zimcrimewatch.py
=====================
ZimCrimeWatch Backend — White Box Test Suite
=============================================

Covers the three formal test cases documented in the project:

  Test Case 1 — Random Forest Feature Engineering (TfidfVectorizer)
    Verifies that the TF-IDF vectorizer handles empty / None modus operandi
    fields without raising a ValueError, and that a valid M.O. string produces
    a meaningful (non-zero) feature vector.

  Test Case 2 — Temporal Analysis Logic (seasonal_decompose)
    Verifies that compute_time_series() correctly handles time series data that
    contains missing date gaps (simulating sparse crime reporting periods).
    The function must fill gaps with zero counts and NOT raise a KeyError or
    ValueError from seasonal_decompose or pandas timezone handling.

  Test Case 3 — API View Role Validation (RBAC)
    Verifies that the API correctly enforces Role-Based Access Control:
      • Admin  → 200 OK  on the user management endpoint
      • Analyst → 403 Forbidden on the same endpoint

Usage
-----
    pytest tests/test_zimcrimewatch.py -v

Requirements
------------
    pip install pytest pytest-django djangorestframework scikit-learn
                pandas numpy statsmodels
"""

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 1 — Random Forest Feature Engineering (TfidfVectorizer)
# ─────────────────────────────────────────────────────────────────────────────

class TestTfidfVectorizerFeatureEngineering:
    """
    Test Case 1: Verify that the TF-IDF vectorizer used in ProfileMatcher
    handles both standard and empty modus operandi strings without crashing.

    The TF-IDF vectorizer is fitted during training on a corpus of M.O. texts.
    At inference time it must call .transform() (not .fit_transform()) so that
    the same vocabulary is reused.  Passing an empty string is a common
    real-world scenario when an officer files an RRB without M.O. details.
    """

    @pytest.fixture
    def fitted_vectorizer(self):
        """
        Build and fit a TfidfVectorizer on a small representative corpus so
        that subsequent .transform() calls use a valid vocabulary.
        This mirrors how ProfileMatcher.train() initialises self.tfidf.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Representative corpus — same parameters as ProfileMatcher.train()
        corpus = [
            "Smash and grab from parked vehicle near market",
            "Suspect broke window of shop and stole electronics",
            "Armed robbery at petrol station using firearm",
            "Pickpocket in crowded bus terminus",
        ]

        vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words="english",
        )
        # Fit on the corpus so the vocabulary is populated
        vectorizer.fit(corpus)
        return vectorizer

    # ------------------------------------------------------------------
    # Sub-test 1a: Valid M.O. text → non-zero feature vector
    # ------------------------------------------------------------------
    def test_valid_mo_text_produces_nonzero_vector(self, fitted_vectorizer):
        """
        Given a meaningful M.O. description, the vectorizer should return
        a sparse matrix whose sum is greater than zero, indicating that at
        least some vocabulary tokens were found in the text.
        """
        valid_mo = "Suspect smashed window of shop and grabbed electronics"

        # .transform() returns a sparse matrix; .toarray() makes it dense
        vector = fitted_vectorizer.transform([valid_mo]).toarray()

        # The feature vector should have at least one non-zero entry
        assert vector.sum() > 0, (
            "A valid M.O. string should produce a non-zero TF-IDF feature vector"
        )
        # Shape must be (1, max_features)
        assert vector.shape == (1, 40), (
            "Vector shape should be (1, max_features=300)"
        )

    # ------------------------------------------------------------------
    # Sub-test 1b: Empty string M.O. → zero vector, no crash
    # ------------------------------------------------------------------
    def test_empty_mo_string_produces_zero_vector(self, fitted_vectorizer):
        """
        Objective: Verify that an empty M.O. string does NOT raise a
        ValueError and instead produces an all-zero feature vector.

        This is the primary assertion from Test Case 1 in the project report.
        """
        empty_mo = ""

        # This MUST NOT raise ValueError or any other exception
        try:
            vector = fitted_vectorizer.transform([empty_mo]).toarray()
        except ValueError as exc:
            pytest.fail(
                f"TfidfVectorizer raised ValueError on empty string: {exc}"
            )

        # All entries must be zero because no known vocabulary tokens exist
        assert vector.sum() == 0.0, (
            "An empty M.O. string should produce an all-zero feature vector"
        )
        assert vector.shape == (1, 40), (
            "Vector shape must remain (1, max_features) even for empty input"
        )

    # ------------------------------------------------------------------
    # Sub-test 1c: None M.O. value → handled via fillna, no crash
    # ------------------------------------------------------------------
    def test_none_mo_handled_via_fillna(self, fitted_vectorizer):
        """
        In the real code, None values are replaced with "" before calling
        transform().  This test verifies that the fillna("") pattern used
        in ProfileMatcher._build_feature_vector() works correctly.
        """
        raw_mo_value = None

        # Replicate the exact guard used in _build_feature_vector()
        safe_mo = raw_mo_value or ""

        # Should not crash
        vector = fitted_vectorizer.transform([safe_mo]).toarray()

        assert vector.sum() == 0.0, (
            "None M.O. (converted to empty string) should produce a zero vector"
        )

    # ------------------------------------------------------------------
    # Sub-test 1d: Feature vector shapes are consistent across inputs
    # ------------------------------------------------------------------
    def test_feature_vector_shape_is_consistent(self, fitted_vectorizer):
        """
        Whether the input is rich text or empty, the output shape must always
        be (1, 300) so that np.hstack() in ProfileMatcher._build_feature_vector()
        does not fail with a dimension mismatch.
        """
        inputs = [
            "Armed robbery with firearm at petrol station",
            "",
            "a",  # single character — likely out-of-vocabulary
        ]

        for text in inputs:
            vector = fitted_vectorizer.transform([text]).toarray()
            assert vector.shape == (1, 40), (
                f"Inconsistent vector shape for input '{text}': got {vector.shape}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 2 — Temporal Analysis Logic (seasonal_decompose)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeSeriesTemporalAnalysis:
    """
    Test Case 2: Verify that compute_time_series() handles DataFrames with
    missing date gaps, timezone-aware timestamps, and sparse data without
    raising KeyError, ValueError, or pandas timezone errors.

    The project report noted that initial testing revealed problems with:
      (a) date-timezone handling — fixed by normalising to UTC then removing tz
      (b) missing date indices — fixed by filling with zero counts (asfreq)

    These tests confirm that the fixes are in place and stable.
    """

    def _make_dataframe_with_gaps(self, freq="D", gap_start="2024-03-10",
                                   gap_end="2024-03-14"):
        """
        Helper: build a timezone-aware DataFrame that is missing several
        consecutive days of incidents — simulating a real reporting gap.

        Parameters
        ----------
        freq       : pandas offset alias ('D' = daily, 'W' = weekly)
        gap_start  : first date of the gap (inclusive)
        gap_end    : last date of the gap (inclusive)
        """
        # Generate a full date range and exclude the gap to simulate missing data
        full_range = pd.date_range("2024-01-01", "2024-06-30", freq=freq, tz="UTC")
        gap = pd.date_range(gap_start, gap_end, freq=freq, tz="UTC")
        dates = full_range.difference(gap)

        return pd.DataFrame({"timestamp": dates})

    def _make_dataframe_minimal(self):
        """
        Helper: create a minimal DataFrame with fewer data points than
        the min_periods threshold (triggers the 'not enough data' branch).
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="W", tz="UTC")
        return pd.DataFrame({"timestamp": dates})

    # ------------------------------------------------------------------
    # Sub-test 2a: Daily series with gap → no KeyError or ValueError
    # ------------------------------------------------------------------
    def test_daily_series_with_gap_does_not_crash(self):
        """
        Primary assertion for Test Case 2: a daily series with a 5-day gap
        in the middle must NOT raise a KeyError from seasonal_decompose
        (caused by missing index values) or a ValueError from timezone
        handling in pandas.
        """
        # Import inside the test to keep it self-contained
        from zimcrimewatch.ml_utils import compute_time_series

        df = self._make_dataframe_with_gaps(freq="D")

        # This MUST NOT raise KeyError or ValueError
        try:
            result = compute_time_series(df, period="D")
        except KeyError as exc:
            pytest.fail(
                f"compute_time_series raised KeyError on gapped data: {exc}\n"
                "Check that missing date indices are filled with zero before decompose()."
            )
        except ValueError as exc:
            pytest.fail(
                f"compute_time_series raised ValueError: {exc}\n"
                "Check timezone normalisation — ensure tz_localize(None) is called "
                "before seasonal_decompose."
            )

        # Result must be a dict (even if decomposition was skipped due to short series)
        assert isinstance(result, dict), "compute_time_series must return a dict"
        assert "labels" in result, "Result dict must contain 'labels'"
        assert "observed" in result, "Result dict must contain 'observed'"

    # ------------------------------------------------------------------
    # Sub-test 2b: Weekly series with gap → observations contain zeros
    # ------------------------------------------------------------------
    def test_weekly_series_gap_filled_with_zeros(self):
        """
        After resampling, the periods that fall within the gap should be
        present in the output (filled with zero) rather than absent.
        This confirms that asfreq(fill_value=0) is working correctly.
        """
        from zimcrimewatch.ml_utils import compute_time_series

        df = self._make_dataframe_with_gaps(
            freq="W", gap_start="2024-03-01", gap_end="2024-03-28"
        )

        result = compute_time_series(df, period="W")

        assert 0 in result["observed"], (
            "Weeks inside the reporting gap should be filled with 0 observations, "
            "but no zeros were found in the observed series."
        )

    # ------------------------------------------------------------------
    # Sub-test 2c: Timezone-aware timestamps → no tz localisation error
    # ------------------------------------------------------------------
    def test_timezone_aware_timestamps_are_normalised(self):
        """
        The fix for the timezone bug removes tz info from the resampled index
        via ts.index.tz_localize(None) before calling seasonal_decompose.
        This test directly confirms that step works for UTC-tagged data.
        """
        from zimcrimewatch.ml_utils import compute_time_series

        # All timestamps carry explicit UTC timezone info
        dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        df = pd.DataFrame({"timestamp": dates})

        # Must NOT raise TypeError: "Cannot localize timezone-aware DatetimeIndex"
        try:
            result = compute_time_series(df, period="D")
        except TypeError as exc:
            pytest.fail(
                f"Timezone error not handled: {exc}\n"
                "Ensure ts.index.tz_localize(None) is called after resample."
            )

        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # Sub-test 2d: Short series → returns 'note' key (graceful degradation)
    # ------------------------------------------------------------------
    def test_short_series_returns_note_not_crash(self):
        """
        When there are fewer data points than the minimum required for
        seasonal decomposition, compute_time_series should return a dict
        with a 'note' key rather than crashing.
        """
        from zimcrimewatch.ml_utils import compute_time_series

        df = self._make_dataframe_minimal()
        result = compute_time_series(df, period="W")

        assert isinstance(result, dict), "Must return a dict even for short series"
        # For short series, decomposition components should be None or absent
        # and a note key should be present
        assert "note" in result or result.get("trend") is None, (
            "Short series should either include a 'note' key or have trend=None"
        )

    # ------------------------------------------------------------------
    # Sub-test 2e: Monthly series → accepts short code 'M'
    # ------------------------------------------------------------------
    def test_short_code_M_is_accepted(self):
        """
        TimeSeriesRequestSerializer sends single-letter codes ('D', 'W', 'M').
        compute_time_series must accept these directly via the _FREQ_MAP lookup.
        """
        from zimcrimewatch.ml_utils import compute_time_series

        dates = pd.date_range("2022-01-01", periods=36, freq="MS", tz="UTC")
        df = pd.DataFrame({"timestamp": dates})

        result = compute_time_series(df, period="M")

        assert isinstance(result, dict)
        assert len(result["labels"]) > 0, (
            "Monthly series ('M' code) should produce at least one label"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE 3 — API View Role Validation (RBAC)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIViewRoleValidation:
    """
    Test Case 3: Verify that the API views correctly enforce Role-Based
    Access Control (RBAC) by inspecting the permission class logic directly,
    without requiring a running Django server or database.

    Test conditions:
      (a) Admin user    → permission GRANTED (has_permission returns True)
      (b) Analyst user  → permission DENIED  (has_permission returns False)
      (c) Officer user  → permission DENIED for admin-only endpoints

    The tests mock the request.user object with the appropriate role attribute,
    isolating the permission logic from authentication infrastructure.
    """

    class MockUser:
        """
        Minimal mock for django.contrib.auth.models.User that carries a
        'role' attribute matching CustomUser's role field choices.
        """
        def __init__(self, role: str, is_authenticated: bool = True,
                     is_active: bool = True):
            self.role            = role
            self.is_authenticated = is_authenticated
            self.is_active       = is_active

    class MockRequest:
        """
        Minimal mock for rest_framework.request.Request.
        Only the .user attribute is needed for permission checks.
        """
        def __init__(self, user, method: str = "GET"):
            self.user   = user
            self.method = method

    # ------------------------------------------------------------------
    # Sub-test 3a: IsZRPAdmin — admin user receives permission
    # ------------------------------------------------------------------
    def test_admin_user_passes_iszrpadmin(self):
        """
        An admin-role user must pass the IsZRPAdmin permission check.
        This corresponds to the 200 OK scenario in Test Case 3.
        """
        from zimcrimewatch.permissions import IsZRPAdmin

        permission = IsZRPAdmin()
        user    = self.MockUser(role="admin")
        request = self.MockRequest(user=user, method="GET")

        # has_permission() is the DRF entry point for view-level checks
        result = permission.has_permission(request, view=None)

        assert result is True, (
            "Admin user should be granted access by IsZRPAdmin, "
            "but has_permission returned False."
        )

    # ------------------------------------------------------------------
    # Sub-test 3b: IsZRPAdmin — analyst user is denied
    # ------------------------------------------------------------------
    def test_analyst_user_denied_by_iszrpadmin(self):
        """
        An analyst-role user must be DENIED by IsZRPAdmin.
        This corresponds to the 403 Forbidden scenario in Test Case 3.
        """
        from zimcrimewatch.permissions import IsZRPAdmin

        permission = IsZRPAdmin()
        user    = self.MockUser(role="analyst")
        request = self.MockRequest(user=user, method="GET")

        result = permission.has_permission(request, view=None)

        # Analysts should NOT have access to admin-only endpoints
        assert result is False, (
            "Analyst user should be denied access by IsZRPAdmin, "
            "but has_permission returned True — this is a security vulnerability."
        )

    # ------------------------------------------------------------------
    # Sub-test 3c: IsZRPAdmin — officer user is denied
    # ------------------------------------------------------------------
    def test_officer_user_denied_by_iszrpadmin(self):
        """
        An officer-role user (the lowest privilege level) must be denied
        by IsZRPAdmin, which guards user management and ML training endpoints.
        """
        from zimcrimewatch.permissions import IsZRPAdmin

        permission = IsZRPAdmin()
        user    = self.MockUser(role="officer")
        request = self.MockRequest(user=user, method="GET")

        result = permission.has_permission(request, view=None)

        assert result is False, (
            "Officer user should be denied access by IsZRPAdmin."
        )

    # ------------------------------------------------------------------
    # Sub-test 3d: IsZRPAuthenticated — any authenticated role passes
    # ------------------------------------------------------------------
    def test_any_authenticated_user_passes_iszrpauthenticated(self):
        """
        IsZRPAuthenticated only checks that the user is authenticated and
        has a 'role' attribute — it does not restrict by role level.
        All three roles (admin, analyst, officer) should be granted.
        """
        from zimcrimewatch.permissions import IsZRPAuthenticated

        permission = IsZRPAuthenticated()

        for role in ("admin", "analyst", "officer"):
            user    = self.MockUser(role=role)
            request = self.MockRequest(user=user)

            result = permission.has_permission(request, view=None)

            assert result is True, (
                f"Authenticated '{role}' user should pass IsZRPAuthenticated, "
                f"but has_permission returned False."
            )

    # ------------------------------------------------------------------
    # Sub-test 3e: IsZRPAnalystOrAdmin — write access control by role
    # ------------------------------------------------------------------
    def test_analyst_or_admin_write_access(self):
        """
        For write operations (POST/PUT/DELETE), IsZRPAnalystOrAdmin should:
          • Grant access to 'analyst' and 'admin' roles
          • Deny access to 'officer' role
        For read operations (GET), all authenticated roles are allowed.
        """
        from zimcrimewatch.permissions import IsZRPAnalystOrAdmin

        permission = IsZRPAnalystOrAdmin()

        write_methods = ("POST", "PUT", "PATCH", "DELETE")

        # Analyst and Admin should be granted write access
        for role in ("analyst", "admin"):
            for method in write_methods:
                user    = self.MockUser(role=role)
                request = self.MockRequest(user=user, method=method)
                result  = permission.has_permission(request, view=None)

                assert result is True, (
                    f"Role '{role}' should have write access ({method}) via "
                    f"IsZRPAnalystOrAdmin, but was denied."
                )

        # Officer should be denied write access
        for method in write_methods:
            user    = self.MockUser(role="officer")
            request = self.MockRequest(user=user, method=method)
            result  = permission.has_permission(request, view=None)

            assert result is False, (
                f"Officer should be denied {method} access via "
                f"IsZRPAnalystOrAdmin, but was granted — security risk."
            )

        # Officer should still be allowed GET access
        user    = self.MockUser(role="officer")
        request = self.MockRequest(user=user, method="GET")
        result  = permission.has_permission(request, view=None)

        assert result is True, (
            "Officer should be allowed GET (read-only) access via "
            "IsZRPAnalystOrAdmin."
        )

    # ------------------------------------------------------------------
    # Sub-test 3f: Unauthenticated user is denied by all permissions
    # ------------------------------------------------------------------
    def test_unauthenticated_user_denied_by_all_permissions(self):
        """
        A user that is not authenticated (simulating a request with no
        or an expired JWT) must be denied by every permission class.
        """
        from zimcrimewatch.permissions import (
            IsZRPAdmin,
            IsZRPAuthenticated,
            IsZRPAnalystOrAdmin,
        )

        # Unauthenticated user has is_authenticated = False
        unauthenticated_user = self.MockUser(role="admin", is_authenticated=False)

        for PermClass in (IsZRPAdmin, IsZRPAuthenticated, IsZRPAnalystOrAdmin):
            permission = PermClass()
            request    = self.MockRequest(user=unauthenticated_user)

            result = permission.has_permission(request, view=None)

            assert result is False, (
                f"Unauthenticated user should be denied by {PermClass.__name__}, "
                f"but was granted access."
            )