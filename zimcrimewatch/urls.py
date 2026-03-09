"""
zimcrimewatch/urls.py
=====================
ZimCrimeWatch — URL Configuration

Maps all API endpoints to their corresponding APIView classes.

Auth endpoints summary
----------------------
POST  api/public/auth/login/          LoginView        — get access + refresh tokens
POST  api/public/auth/token/refresh/  TokenRefreshView — exchange refresh → new access
POST  api/public/auth/logout/         LogoutView       — blacklist refresh token

Serial Crime Linkage endpoints  ← NEW
--------------------------------------
POST  api/zrp/analytics/serial-linkage/train/            — train DBSCAN cluster model
POST  api/zrp/analytics/serial-linkage/cluster/          — get cluster assignments
POST  api/zrp/analytics/serial-linkage/link-probability/ — pairwise link probability
"""
from django.urls import path
from . import (views, csv_upload_view)  # Import views and the new CSV upload view


app_name = "zimcrimewatch"

urlpatterns = [

    # ──────────────────────────────────────────────────────────────────────────
    # Authentication  (used by both Flutter app and React dashboard)
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "public/auth/login/",
        views.LoginView.as_view(),
        name="auth-login",
    ),
    path(
        "public/auth/token/refresh/",
        views.TokenRefreshView.as_view(),
        name="auth-token-refresh",
    ),
    path(
        "public/auth/logout/",
        views.LogoutView.as_view(),
        name="auth-logout",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Public  (Flutter mobile app — no authentication required)
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "public/crimes/",
        views.PublicCrimeMapView.as_view(),
        name="public-crimes",
    ),
    path(
        "public/crime-types/",
        views.PublicCrimeTypeListView.as_view(),
        name="public-crime-types",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # ZRP Dashboard — Incident CRUD  (React web app, JWT required)
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/incidents/",
        views.IncidentListCreateView.as_view(),
        name="incident-list",
    ),
    path(
        "zrp/incidents/<int:pk>/",
        views.IncidentDetailView.as_view(),
        name="incident-detail",
    ),
    # ProfileMatcher similar-cases endpoint — uses the fixed find_similar()
    path(
        "zrp/incidents/<int:pk>/similar/",
        views.IncidentSimilarCasesView.as_view(),
        name="incident-similar",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # ZRP Dashboard — Crime Type CRUD
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/crime-types/",
        views.CrimeTypeListCreateView.as_view(),
        name="crime-type-list",
    ),
    path(
        "zrp/crime-types/<int:pk>/",
        views.CrimeTypeDetailView.as_view(),
        name="crime-type-detail",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # ZRP Dashboard — KPI summary
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/dashboard/summary/",
        views.DashboardSummaryView.as_view(),
        name="dashboard-summary",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Analytics modules  (analyst + admin only)
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/analytics/heatmap/",
        views.HeatmapView.as_view(),
        name="analytics-heatmap",
    ),
    path(
        "zrp/analytics/timeseries/",
        views.TimeSeriesView.as_view(),
        name="analytics-timeseries",
    ),
    path(
        "zrp/analytics/hotspots/",
        views.HotspotView.as_view(),
        name="analytics-hotspots",
    ),
    path(
        "zrp/analytics/profile-match/",
        views.ProfileMatchView.as_view(),
        name="analytics-profile-match",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Serial Crime Linkage  ← NEW
    # Integrates serial_crime_linkage.py into the backend API.
    # ──────────────────────────────────────────────────────────────────────────

    # Train the DBSCAN serial-linkage model on all current incidents.
    path(
        "zrp/analytics/serial-linkage/train/",
        views.SerialLinkageTrainView.as_view(),
        name="serial-linkage-train",
    ),

    # Return cluster assignments for all incidents (or a filtered subset).
    path(
        "zrp/analytics/serial-linkage/cluster/",
        views.SerialLinkageClusterView.as_view(),
        name="serial-linkage-cluster",
    ),

    # Compute pairwise link probability between two specific incidents.
    path(
        "zrp/analytics/serial-linkage/link-probability/",
        views.SerialLinkageProbabilityView.as_view(),
        name="serial-linkage-probability",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Admin — User management
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/users/",
        views.UserListCreateView.as_view(),
        name="user-list",
    ),
    path(
        "zrp/users/<int:pk>/",
        views.UserDetailView.as_view(),
        name="user-detail",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Admin — ML model training trigger  (ProfileMatcher only)
    # For serial linkage training use /analytics/serial-linkage/train/ instead.
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/ml/train/",
        views.MLTrainView.as_view(),
        name="ml-train",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Admin — Bulk CSV data upload  (fulfils ADM-03 requirement)
    # ──────────────────────────────────────────────────────────────────────────
    # POST  /api/zrp/data/upload-csv/
    #
    # Accepts multipart/form-data with one field:
    #   file  — a .csv matching the zimcrime_watch_dataset schema
    #
    # Returns: { "created": N, "skipped": N, "errors": [...] }
    # ──────────────────────────────────────────────────────────────────────────
    path(
        "zrp/data/upload-csv/",
        csv_upload_view.CSVUploadView.as_view(),
        name="data-upload-csv",
    ),
]
