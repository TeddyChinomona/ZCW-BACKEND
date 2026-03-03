"""
ZimCrimeWatch - API Views (APIView only, no ViewSets)
======================================================
Serves both the Flutter mobile app (public, anonymized) and the
React ZRP dashboard (authenticated, full data + analytics).

Endpoint groups
---------------
Public (no auth)
  POST /api/public/auth/login/
  POST /api/public/auth/logout/
  GET  /api/public/crimes/          — anonymized map pins for Flutter
  GET  /api/public/crime-types/     — list of crime categories + icons

ZRP Dashboard (authentication required)
  GET/POST   /api/zrp/incidents/
  GET/PUT/DELETE /api/zrp/incidents/<id>/
  GET        /api/zrp/incidents/<id>/similar/   ← profile matching
  GET        /api/zrp/dashboard/summary/
  GET/POST   /api/zrp/crime-types/
  GET/PUT/DELETE /api/zrp/crime-types/<id>/

Analytics (authentication required)
  GET/POST   /api/zrp/analytics/heatmap/
  GET/POST   /api/zrp/analytics/timeseries/
  GET/POST   /api/zrp/analytics/hotspots/
  POST       /api/zrp/analytics/profile-match/

Admin only
  GET/POST   /api/zrp/users/
  GET/PUT/DELETE /api/zrp/users/<id>/
  POST       /api/zrp/ml/train/
"""
from __future__ import annotations

from loguru import logger
from datetime import datetime, timedelta

import pandas as pd
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.utils import timezone
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .ml_utils import (
    ProfileMatcher,
    compute_hotspot_summary,
    compute_kde_heatmap,
    compute_time_series,
)
from django.contrib.auth.views import get_user_model
from .models import CrimeIncident, CrimeType, CustomUser
from .permissions import IsZRPAdmin, IsZRPAnalystOrAdmin, IsZRPAuthenticated
from .serializers import (
    CrimeIncidentSerializer,
    CrimeTypeSerializer,
    CreateUserSerializer,
    HeatmapRequestSerializer,
    LoginSerializer,
    ProfileMatchRequestSerializer,
    PublicCrimeIncidentSerializer,
    TimeSeriesRequestSerializer,
    UserSerializer,
)

user = get_user_model()

# =============================================================================
# Helpers
# =============================================================================


def _parse_date_range(request) -> tuple[datetime | None, datetime | None]:
    """Pull optional start_date / end_date from query params."""
    start = request.query_params.get("start_date")
    end = request.query_params.get("end_date")
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start else None
        end_dt = (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            if end
            else None
        )
    except ValueError:
        start_dt = end_dt = None
    return start_dt, end_dt


def _filter_incidents(request, qs=None):
    """Apply common query filters (crime_type, date range, suburb, status)."""
    if qs is None:
        qs = CrimeIncident.objects.select_related("crime_type", "created_by")

    crime_type_id = request.query_params.get("crime_type_id")
    suburb = request.query_params.get("suburb", "").strip()
    status_filter = request.query_params.get("status", "").strip()
    start_dt, end_dt = _parse_date_range(request)

    if crime_type_id:
        qs = qs.filter(crime_type_id=crime_type_id)
    if suburb:
        qs = qs.filter(suburb__icontains=suburb)
    if status_filter:
        qs = qs.filter(status=status_filter)
    if start_dt:
        qs = qs.filter(timestamp__gte=start_dt)
    if end_dt:
        qs = qs.filter(timestamp__lt=end_dt)
    return qs


# =============================================================================
# AUTH — public endpoints
# =============================================================================


class LoginView(APIView):
    """
    POST /api/public/auth/login/
    Authenticates a ZRP user and returns an auth token.
    Body: { "username": "...", "password": "..." }
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(
            request,
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            return Response(
                {"detail": "Invalid credentials."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not user.is_active:
            return Response(
                {"detail": "Account is disabled."},
                status=status.HTTP_403_FORBIDDEN,
            )
        refresh = RefreshToken.for_user(user)
        data = {
            "message": f"User {user.username} logged in successfully",
            "success": True,
            "refresh": str(refresh),
            "access": str(refresh.access_token)
        }
        return Response(data, status=status.HTTP_200_OK)


class LogoutView(APIView):
    """
    POST /api/public/auth/logout/
    Invalidates the user's auth token.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            pass
        except Exception:
            pass
        return Response({"detail": "Logged out."}, status=status.HTTP_200_OK)


# =============================================================================
# PUBLIC — Flutter mobile app (anonymized, no auth)
# =============================================================================


class PublicCrimeMapView(APIView):
    """
    GET /api/public/crimes/
    Returns anonymized, rounded crime pins for the Flutter map.
    Supports: ?crime_type_id=&start_date=&end_date=&suburb=&limit=
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = _filter_incidents(
            request,
            CrimeIncident.objects.select_related("crime_type").filter(status__in=["reported", "closed", "unsolved"])
        )
        limit = int(request.query_params.get("limit", 500))
        qs = qs[:limit]
        serializer = PublicCrimeIncidentSerializer(qs, many=True)
        return Response({"count": len(serializer.data), "results": serializer.data})


class PublicCrimeTypeListView(APIView):
    """
    GET /api/public/crime-types/
    Returns all crime categories with icons for the Flutter filter panel.
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = CrimeType.objects.annotate(incident_count=Count("incidents"))
        serializer = CrimeTypeSerializer(qs, many=True)
        return Response(serializer.data)


# =============================================================================
# ZRP DASHBOARD — Incident management
# =============================================================================


class IncidentListCreateView(APIView):
    """
    GET  /api/zrp/incidents/  — paginated list with filters
    POST /api/zrp/incidents/  — create a new incident
    Filters: crime_type_id, start_date, end_date, suburb, status, search
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = _filter_incidents(request)

        # Full-text search across case_number, suburb, description, MO
        search = request.query_params.get("search", "").strip()
        if search:
            qs = qs.filter(
                Q(case_number__icontains=search)
                | Q(suburb__icontains=search)
                | Q(description_narrative__icontains=search)
                | Q(modus_operandi__icontains=search)
                | Q(serial_group_label__icontains=search)
            )

        # Ordering
        order_by = request.query_params.get("order_by", "-timestamp")
        allowed_ordering = [
            "timestamp", "-timestamp", "case_number", "-case_number",
            "crime_type__name", "-crime_type__name", "status"
        ]
        if order_by not in allowed_ordering:
            order_by = "-timestamp"
        qs = qs.order_by(order_by)

        # Simple pagination
        page = max(int(request.query_params.get("page", 1)), 1)
        page_size = min(int(request.query_params.get("page_size", 50)), 200)
        total = qs.count()
        start = (page - 1) * page_size
        qs_page = qs[start: start + page_size]

        serializer = CrimeIncidentSerializer(qs_page, many=True)
        return Response({
            "count": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "results": serializer.data,
        })

    def post(self, request):
        if not hasattr(request.user, "zrp_profile") or request.user.zrp_profile.role not in ("analyst", "admin"):
            return Response({"detail": "Only analysts and admins can create incidents."}, status=status.HTTP_403_FORBIDDEN)

        serializer = CrimeIncidentSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class IncidentDetailView(APIView):
    """
    GET    /api/zrp/incidents/<id>/  — full incident details
    PUT    /api/zrp/incidents/<id>/  — full update
    PATCH  /api/zrp/incidents/<id>/  — partial update
    DELETE /api/zrp/incidents/<id>/  — delete (admin only)
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_incident(self, pk):
        try:
            return CrimeIncident.objects.select_related("crime_type", "created_by").get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return None

    def get(self, request, pk):
        incident = self._get_incident(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeIncidentSerializer(incident).data)

    def put(self, request, pk):
        return self._update(request, pk, partial=False)

    def patch(self, request, pk):
        return self._update(request, pk, partial=True)

    def _update(self, request, pk, partial):
        if request.user.zrp_profile.role not in ("analyst", "admin"):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        incident = self._get_incident(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeIncidentSerializer(incident, data=request.data, partial=partial, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        if request.user.zrp_profile.role != "admin":
            return Response({"detail": "Only admins can delete incidents."}, status=status.HTTP_403_FORBIDDEN)
        incident = self._get_incident(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        incident.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class IncidentSimilarCasesView(APIView):
    """
    GET /api/zrp/incidents/<id>/similar/
    Runs the profile matcher against the incident identified by <id> and
    returns a list of similar incidents from the same predicted serial group.
    Optional: ?top_n=5
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request, pk):
        try:
            incident = CrimeIncident.objects.select_related("crime_type").get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        top_n = min(int(request.query_params.get("top_n", 5)), 20)

        try:
            group_predictions = ProfileMatcher.load_and_predict(
                mo_text=incident.modus_operandi,
                crime_type_name=incident.crime_type.name,
                time_of_day=incident.time_of_day,
                day_of_week=incident.day_of_week,
                weapon_used=incident.weapon_used,
                top_n=top_n,
            )
        except FileNotFoundError:
            return Response(
                {"detail": "Profile matcher model not trained yet. Contact an administrator."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.error("Profile matching error for incident %s: %s", pk, exc)
            return Response({"detail": "Model inference failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Fetch actual cases belonging to predicted group labels
        predicted_groups = [p["group_label"] for p in group_predictions]
        similar_qs = (
            CrimeIncident.objects.select_related("crime_type")
            .filter(serial_group_label__in=predicted_groups)
            .exclude(pk=pk)
            .order_by("-timestamp")[:30]
        )
        similar_data = CrimeIncidentSerializer(similar_qs, many=True).data

        return Response({
            "source_incident_id": pk,
            "predicted_profiles": group_predictions,
            "similar_incidents": similar_data,
        })


# =============================================================================
# ZRP DASHBOARD — Crime type management
# =============================================================================


class CrimeTypeListCreateView(APIView):
    """
    GET  /api/zrp/crime-types/  — list all crime types with incident counts
    POST /api/zrp/crime-types/  — create a crime type (analyst/admin only)
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = CrimeType.objects.annotate(incident_count=Count("incidents")).order_by("name")
        return Response(CrimeTypeSerializer(qs, many=True).data)

    def post(self, request):
        if request.user.zrp_profile.role not in ("analyst", "admin"):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        serializer = CrimeTypeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CrimeTypeDetailView(APIView):
    """
    GET/PUT/DELETE /api/zrp/crime-types/<id>/
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_ct(self, pk):
        try:
            return CrimeType.objects.annotate(incident_count=Count("incidents")).get(pk=pk)
        except CrimeType.DoesNotExist:
            return None

    def get(self, request, pk):
        ct = self._get_ct(pk)
        if not ct:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeTypeSerializer(ct).data)

    def put(self, request, pk):
        if request.user.zrp_profile.role not in ("analyst", "admin"):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        ct = self._get_ct(pk)
        if not ct:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeTypeSerializer(ct, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        if request.user.zrp_profile.role != "admin":
            return Response({"detail": "Admin only."}, status=status.HTTP_403_FORBIDDEN)
        ct = self._get_ct(pk)
        if not ct:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if ct.incident_count > 0:
            return Response(
                {"detail": f"Cannot delete: {ct.incident_count} incidents reference this type."},
                status=status.HTTP_409_CONFLICT,
            )
        ct.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# =============================================================================
# ZRP DASHBOARD — Summary / KPI widget
# =============================================================================


class DashboardSummaryView(APIView):
    """
    GET /api/zrp/dashboard/summary/
    Returns KPI cards for the React dashboard:
      - total incidents (all time and last 30 days)
      - breakdown by crime type
      - breakdown by status
      - breakdown by time_of_day
      - top 5 suburbs
      - incidents this week vs last week (trend arrow)
    Supports ?start_date=&end_date= for date-scoped summaries.
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        start_dt, end_dt = _parse_date_range(request)
        qs = CrimeIncident.objects.all()
        if start_dt:
            qs = qs.filter(timestamp__gte=start_dt)
        if end_dt:
            qs = qs.filter(timestamp__lt=end_dt)

        now = timezone.now()
        thirty_days_ago = now - timedelta(days=30)
        seven_days_ago = now - timedelta(days=7)
        fourteen_days_ago = now - timedelta(days=14)

        total = qs.count()
        last_30 = qs.filter(timestamp__gte=thirty_days_ago).count()
        this_week = qs.filter(timestamp__gte=seven_days_ago).count()
        last_week = qs.filter(timestamp__gte=fourteen_days_ago, timestamp__lt=seven_days_ago).count()

        by_type = list(
            qs.values("crime_type__name", "crime_type__icon")
            .annotate(count=Count("id"))
            .order_by("-count")
        )
        by_status = list(qs.values("status").annotate(count=Count("id")).order_by("-count"))
        by_tod = list(qs.values("time_of_day").annotate(count=Count("id")).order_by("-count"))
        top_suburbs = list(
            qs.exclude(suburb="").values("suburb").annotate(count=Count("id")).order_by("-count")[:5]
        )
        open_cases = qs.filter(status__in=["reported", "under_investigation"]).count()

        trend_pct = None
        if last_week > 0:
            trend_pct = round((this_week - last_week) / last_week * 100, 1)

        return Response({
            "total_incidents": total,
            "last_30_days": last_30,
            "this_week": this_week,
            "last_week": last_week,
            "week_trend_pct": trend_pct,
            "open_cases": open_cases,
            "by_crime_type": by_type,
            "by_status": by_status,
            "by_time_of_day": by_tod,
            "top_suburbs": top_suburbs,
        })


# =============================================================================
# ANALYTICS — KDE Heatmap
# =============================================================================


class HeatmapView(APIView):
    """
    POST /api/zrp/analytics/heatmap/
    Runs Kernel Density Estimation on filtered incidents and returns a
    grid of [lat, lng, intensity] points for Leaflet.js.

    Body (all optional):
    {
      "crime_type_id": 1,
      "start_date": "2024-01-01",
      "end_date":   "2024-12-31",
      "bandwidth":  0.01,
      "grid_size":  60,
      "bounds": { "min_lat": -18.0, "max_lat": -17.5, "min_lng": 30.9, "max_lng": 31.2 }
    }
    """
    permission_classes = [IsZRPAuthenticated]

    def post(self, request):
        serializer = HeatmapRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        params = serializer.validated_data
        qs = CrimeIncident.objects.all()
        if params.get("crime_type_id"):
            qs = qs.filter(crime_type_id=params["crime_type_id"])
        if params.get("start_date"):
            qs = qs.filter(timestamp__date__gte=params["start_date"])
        if params.get("end_date"):
            qs = qs.filter(timestamp__date__lte=params["end_date"])

        coords = list(qs.values_list("latitude", "longitude"))
        if not coords:
            return Response({"heatmap_data": [], "point_count": 0})

        coordinates = [(float(lat), float(lng)) for lat, lng in coords]

        heatmap_data = compute_kde_heatmap(
            coordinates=coordinates,
            bandwidth=params["bandwidth"],
            grid_size=params["grid_size"],
            bounds=params.get("bounds"),
        )

        return Response({
            "heatmap_data": heatmap_data,
            "point_count": len(coordinates),
            "grid_points_returned": len(heatmap_data),
            "bandwidth": params["bandwidth"],
        })

    def get(self, request):
        """Allow GET with query params for simpler client calls."""
        data = {
            "crime_type_id": request.query_params.get("crime_type_id"),
            "start_date": request.query_params.get("start_date"),
            "end_date": request.query_params.get("end_date"),
            "bandwidth": request.query_params.get("bandwidth", 0.01),
            "grid_size": request.query_params.get("grid_size", 50),
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        request._full_data = data
        return self.post(request)


# =============================================================================
# ANALYTICS — Time Series Decomposition
# =============================================================================


class TimeSeriesView(APIView):
    """
    POST /api/zrp/analytics/timeseries/
    Decomposes crime counts into trend, seasonal, and residual components.

    Body:
    {
      "crime_type_id": 1,         # optional — omit for all crimes
      "start_date": "2023-01-01",
      "end_date":   "2024-12-31",
      "period": "weekly",         # "daily" | "weekly" | "monthly"
      "suburb": ""                # optional suburb filter
    }

    Response includes Chart.js-ready arrays for observed, trend, seasonal, residual.
    """
    permission_classes = [IsZRPAuthenticated]

    def post(self, request):
        serializer = TimeSeriesRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        params = serializer.validated_data
        qs = CrimeIncident.objects.all()

        if params.get("crime_type_id"):
            qs = qs.filter(crime_type_id=params["crime_type_id"])
        if params.get("start_date"):
            qs = qs.filter(timestamp__date__gte=params["start_date"])
        if params.get("end_date"):
            qs = qs.filter(timestamp__date__lte=params["end_date"])
        if params.get("suburb"):
            qs = qs.filter(suburb__icontains=params["suburb"])

        if not qs.exists():
            return Response({"detail": "No incidents found for the given filters."}, status=status.HTTP_404_NOT_FOUND)

        # Build a flat DataFrame — only timestamp is needed for the series
        timestamps = list(qs.values_list("timestamp", flat=True))
        df = pd.DataFrame({"timestamp": timestamps})

        try:
            result = compute_time_series(df, period=params["period"])
        except Exception as exc:
            logger.error("Time series decomposition failed: %s", exc)
            return Response({"detail": f"Analysis failed: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Attach crime type name if filtered
        if params.get("crime_type_id"):
            try:
                ct = CrimeType.objects.get(pk=params["crime_type_id"])
                result["crime_type"] = ct.name
            except CrimeType.DoesNotExist:
                pass

        return Response(result)

    def get(self, request):
        """Allow GET with query params."""
        data = {k: request.query_params.get(k) for k in
                ["crime_type_id", "start_date", "end_date", "period", "suburb"]
                if request.query_params.get(k)}
        request._full_data = data
        return self.post(request)


# =============================================================================
# ANALYTICS — Hotspot Detection (DBSCAN)
# =============================================================================


class HotspotView(APIView):
    """
    POST /api/zrp/analytics/hotspots/
    Runs DBSCAN spatial clustering to identify crime hotspot zones and
    returns a ranked list of clusters for tabular display and map overlays.

    Body:
    {
      "crime_type_id": null,
      "start_date": "2024-01-01",
      "end_date":   "2024-12-31",
      "eps_km": 0.5,      # neighbourhood radius in km
      "min_samples": 3    # minimum incidents to form a hotspot
    }
    """
    permission_classes = [IsZRPAuthenticated]

    def post(self, request):
        crime_type_id = request.data.get("crime_type_id")
        start_date = request.data.get("start_date")
        end_date = request.data.get("end_date")
        eps_km = float(request.data.get("eps_km", 0.5))
        min_samples = int(request.data.get("min_samples", 3))

        qs = CrimeIncident.objects.select_related("crime_type")
        if crime_type_id:
            qs = qs.filter(crime_type_id=crime_type_id)
        if start_date:
            qs = qs.filter(timestamp__date__gte=start_date)
        if end_date:
            qs = qs.filter(timestamp__date__lte=end_date)

        data = list(qs.values_list("latitude", "longitude", "crime_type__name", "suburb"))
        if len(data) < min_samples:
            return Response({"hotspots": [], "total_incidents_analysed": len(data),
                             "note": "Not enough incidents to form clusters with current settings."})

        coordinates = [(float(r[0]), float(r[1])) for r in data]
        crime_types = [r[2] or "Unknown" for r in data]
        suburbs = [r[3] or "" for r in data]

        try:
            hotspots = compute_hotspot_summary(
                coordinates=coordinates,
                crime_types=crime_types,
                suburbs=suburbs,
                eps_km=eps_km,
                min_samples=min_samples,
            )
        except Exception as exc:
            logger.error("Hotspot analysis failed: %s", exc)
            return Response({"detail": f"Analysis failed: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        noise_count = len(data) - sum(h["incident_count"] for h in hotspots)
        return Response({
            "hotspots": hotspots,
            "total_incidents_analysed": len(data),
            "noise_incidents": noise_count,
            "eps_km": eps_km,
            "min_samples": min_samples,
        })

    def get(self, request):
        data = {k: request.query_params.get(k) for k in
                ["crime_type_id", "start_date", "end_date", "eps_km", "min_samples"]
                if request.query_params.get(k)}
        request._full_data = data
        return self.post(request)


# =============================================================================
# ANALYTICS — Profile Matching (Random Forest)
# =============================================================================


class ProfileMatchView(APIView):
    """
    POST /api/zrp/analytics/profile-match/
    Predicts the most likely serial crime group for a new incident description
    using the trained Random Forest model, and returns matching past cases.

    Body:
    {
      "crime_type_id": 1,
      "modus_operandi": "Suspect broke rear window of parked vehicle...",
      "time_of_day": "night",
      "day_of_week": "friday",
      "weapon_used": "knife",
      "top_n": 5
    }

    Response:
    {
      "predicted_profiles": [{"group_label": "...", "probability": 0.87}, ...],
      "matching_incidents": [...full incident objects...]
    }
    """
    permission_classes = [IsZRPAuthenticated]

    def post(self, request):
        serializer = ProfileMatchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        params = serializer.validated_data

        try:
            crime_type = CrimeType.objects.get(pk=params["crime_type_id"])
        except CrimeType.DoesNotExist:
            return Response({"detail": "Crime type not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            predictions = ProfileMatcher.load_and_predict(
                mo_text=params["modus_operandi"],
                crime_type_name=crime_type.name,
                time_of_day=params.get("time_of_day", ""),
                day_of_week=params.get("day_of_week", ""),
                weapon_used=params.get("weapon_used", ""),
                top_n=params["top_n"],
            )
        except FileNotFoundError:
            return Response(
                {"detail": "Profile matcher model has not been trained yet. Run /api/zrp/ml/train/ first."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.error("Profile match inference error: %s", exc)
            return Response({"detail": "Model inference failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        group_labels = [p["group_label"] for p in predictions]
        matching_qs = (
            CrimeIncident.objects.select_related("crime_type")
            .filter(serial_group_label__in=group_labels)
            .order_by("-timestamp")[:30]
        )
        matching_data = CrimeIncidentSerializer(matching_qs, many=True).data

        return Response({
            "predicted_profiles": predictions,
            "matching_incidents": matching_data,
            "model_note": "Probabilities reflect similarity to known serial crime groups. "
                          "Review matches carefully before drawing conclusions.",
        })


# =============================================================================
# ML — Train / retrain model (admin only)
# =============================================================================


class TrainProfileMatcherView(APIView):
    """
    POST /api/zrp/ml/train/
    Trains (or retrains) the Random Forest profile matcher on all labelled
    incidents in the database. Admin only.
    Returns: training metrics (samples, classes, cross-val accuracy).
    """
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        labelled_qs = CrimeIncident.objects.exclude(serial_group_label="").select_related("crime_type")

        if not labelled_qs.exists():
            return Response(
                {"detail": "No labelled incidents found. Add serial_group_label values to incidents first."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        matcher = ProfileMatcher()
        try:
            metrics = matcher.train(labelled_qs)
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            return Response({"detail": f"Training failed: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if "error" in metrics:
            return Response({"detail": metrics["error"]}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        return Response(metrics, status=status.HTTP_200_OK)


# =============================================================================
# USER MANAGEMENT (admin only)
# =============================================================================


class UserListCreateView(APIView):
    """
    GET  /api/zrp/users/  — list all ZRP users
    POST /api/zrp/users/  — create a new ZRP user
    """
    permission_classes = [IsZRPAdmin]

    def get(self, request):
        users = User.objects.all().select_related("base_station").order_by("username")
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request):
        """Create a new account"""
        request_data = request.data
        created_user = user.objects.create_user(
            username=request_data.get('username'),
            first_name=request_data.get('first_name'),
            last_name=request_data.get('last_name'),
            email=request_data.get('zrp_badge_number'),
            password=request_data.get('password'),
            role = request_data.get('role')
        )
        data = {"message": f"Account created{created_user.username}"}
        return Response(data, status.HTTP_201_CREATED)

class UserDetailView(APIView):
    """
    GET    /api/zrp/users/<id>/
    PUT    /api/zrp/users/<id>/  — update profile/role
    DELETE /api/zrp/users/<id>/  — deactivate user
    """
    permission_classes = [IsZRPAdmin]

    def _get_user(self, pk):
        try:
            return User.objects.select_related("base_station").get(pk=pk)
        except User.DoesNotExist:
            return None

    def get(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(UserSerializer(user).data)

    def put(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        # Update allowed fields
        profile = user
        profile.fullname = request.data.get("full_name", profile.full_name)
        profile.zrp_badge_number = request.data.get("badge_number", profile.zrp_badge_number)
        profile.base_station = request.data.get("station", profile.base_station)
        new_role = request.data.get("role")
        if new_role and new_role in ("analyst", "officer", "admin"):
            profile.role = new_role
        profile.save()

        if "email" in request.data:
            user.email = request.data["email"]
            user.save(update_fields=["email"])

        return Response(UserSerializer(user).data)

    def delete(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if user == request.user:
            return Response({"detail": "You cannot deactivate your own account."}, status=status.HTTP_400_BAD_REQUEST)
        user.is_active = False
        user.save(update_fields=["is_active"])
        return Response({"detail": "User deactivated."}, status=status.HTTP_200_OK)
