"""
zimcrimewatch/urls.py
=====================
ZimCrimeWatch — URL Configuration

All endpoints map directly to APIView subclasses.

Auth endpoints
--------------
POST  /api/public/auth/login/             LoginView               — get access + refresh tokens
POST  /api/public/auth/register/          RegisterView            — NEW: self-register as officer
POST  /api/public/auth/token/refresh/     TokenRefreshView        — exchange refresh → new access
POST  /api/public/auth/logout/            LogoutView              — blacklist refresh token
POST  /api/public/auth/forgot-password/   ForgotPasswordView      — NEW: request reset token
POST  /api/public/auth/reset-password/    ResetPasswordView       — NEW: set new password with token
POST  /api/zrp/auth/change-password/      ChangePasswordView      — NEW: authenticated change

Public endpoints (no auth required)
-------------------------------------
GET   /api/public/crimes/                 PublicCrimeMapView
GET   /api/public/crime-types/            PublicCrimeTypeListView

ZRP Dashboard (auth required)
------------------------------
GET   /api/zrp/incidents/                 IncidentListCreateView
POST  /api/zrp/incidents/                 IncidentListCreateView
GET   /api/zrp/incidents/<id>/            IncidentDetailView
PUT   /api/zrp/incidents/<id>/            IncidentDetailView
DEL   /api/zrp/incidents/<id>/            IncidentDetailView
GET   /api/zrp/incidents/<id>/similar/    IncidentSimilarCasesView
GET   /api/zrp/dashboard/summary/         DashboardSummaryView
GET   /api/zrp/crime-types/               CrimeTypeListCreateView
POST  /api/zrp/crime-types/               CrimeTypeListCreateView
GET   /api/zrp/crime-types/<id>/          CrimeTypeDetailView
PUT   /api/zrp/crime-types/<id>/          CrimeTypeDetailView
DEL   /api/zrp/crime-types/<id>/          CrimeTypeDetailView

Analytics (analyst + admin required)
--------------------------------------
GET   /api/zrp/analytics/heatmap/
POST  /api/zrp/analytics/heatmap/
GET   /api/zrp/analytics/timeseries/
POST  /api/zrp/analytics/timeseries/
GET   /api/zrp/analytics/hotspots/
POST  /api/zrp/analytics/hotspots/
POST  /api/zrp/analytics/profile-match/

Serial Crime Linkage (analyst + admin required)
------------------------------------------------
POST  /api/zrp/analytics/serial-linkage/train/
POST  /api/zrp/analytics/serial-linkage/cluster/
POST  /api/zrp/analytics/serial-linkage/link-probability/

Admin only
-----------
GET   /api/zrp/users/                     UserListCreateView
POST  /api/zrp/users/                     UserListCreateView
GET   /api/zrp/users/<id>/                UserDetailView
PUT   /api/zrp/users/<id>/                UserDetailView
DEL   /api/zrp/users/<id>/                UserDetailView
POST  /api/zrp/ml/train/                  MLTrainView
POST  /api/zrp/data/upload-csv/           CSVUploadView
"""
from django.urls import path

from . import views, csv_upload_view

app_name = "zimcrimewatch"

urlpatterns = [

    # ── Authentication ────────────────────────────────────────────────────────
    path(
        "public/auth/login/",
        views.LoginView.as_view(),
        name="auth-login",
    ),
    path(
        "public/auth/register/",
        views.RegisterView.as_view(),
        name="auth-register",
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
    path(
        "public/auth/forgot-password/",
        views.ForgotPasswordView.as_view(),
        name="auth-forgot-password",
    ),
    path(
        "public/auth/reset-password/",
        views.ResetPasswordView.as_view(),
        name="auth-reset-password",
    ),
    # Authenticated password change (different prefix — requires JWT)
    path(
        "zrp/auth/change-password/",
        views.ChangePasswordView.as_view(),
        name="auth-change-password",
    ),

    # ── Public (Flutter mobile app — no auth required) ────────────────────────
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

    # ── ZRP Dashboard — Incident CRUD ─────────────────────────────────────────
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
    path(
        "zrp/incidents/<int:pk>/similar/",
        views.IncidentSimilarCasesView.as_view(),
        name="incident-similar",
    ),

    # ── ZRP Dashboard — KPI summary ───────────────────────────────────────────
    path(
        "zrp/dashboard/summary/",
        views.DashboardSummaryView.as_view(),
        name="dashboard-summary",
    ),

    # ── ZRP Dashboard — Crime Type CRUD ──────────────────────────────────────
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

    # ── Analytics modules (analyst + admin only) ──────────────────────────────
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

    # ── Serial Crime Linkage ──────────────────────────────────────────────────
    path(
        "zrp/analytics/serial-linkage/train/",
        views.SerialLinkageTrainView.as_view(),
        name="serial-linkage-train",
    ),
    path(
        "zrp/analytics/serial-linkage/cluster/",
        views.SerialLinkageClusterView.as_view(),
        name="serial-linkage-cluster",
    ),
    path(
        "zrp/analytics/serial-linkage/link-probability/",
        views.SerialLinkageProbabilityView.as_view(),
        name="serial-linkage-probability",
    ),

    # ── Admin — User management ───────────────────────────────────────────────
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

    # ── Admin — ML model training ─────────────────────────────────────────────
    path(
        "zrp/ml/train/",
        views.MLTrainView.as_view(),
        name="ml-train",
    ),

    # ── Admin — Bulk CSV data upload (fulfils ADM-03) ─────────────────────────
    path(
        "zrp/data/upload-csv/",
        csv_upload_view.CSVUploadView.as_view(),
        name="data-upload-csv",
    ),
]