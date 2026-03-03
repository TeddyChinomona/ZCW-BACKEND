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

from datetime import datetime, timedelta

import pandas as pd
from django.contrib.auth import authenticate
from django.db.models import Count, Q
from django.utils import timezone
from loguru import logger
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError, InvalidToken
from rest_framework_simplejwt.tokens import RefreshToken

from .ml_utils import (
    ProfileMatcher,
    compute_hotspot_summary,
    compute_kde_heatmap,
    compute_time_series,
)
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
    TokenRefreshSerializer,
    UserSerializer,
)



# =============================================================================
# Helpers
# =============================================================================

def _parse_date_range(request) -> tuple[datetime | None, datetime | None]:
    """Pull optional start_date / end_date from query params."""
    start = request.query_params.get("start_date")
    end   = request.query_params.get("end_date")
    try:
        start_dt = (
            datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if start else None
        )
        end_dt = (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            + timedelta(days=1)
            if end else None
        )
    except ValueError:
        start_dt = end_dt = None
    return start_dt, end_dt


def _filter_incidents(request, qs=None):
    """Apply common query filters (crime_type, date range, suburb, status)."""
    if qs is None:
        qs = CrimeIncident.objects.select_related("crime_type", "created_by")

    crime_type_id = request.query_params.get("crime_type_id")
    suburb        = request.query_params.get("suburb", "").strip()
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
# AUTH
# =============================================================================

class LoginView(APIView):
    """
    POST /api/public/auth/login/

    Authenticate a ZRP officer and return a SimpleJWT access + refresh token pair.

    Request body
    ------------
    {
        "zrp_badge_number": "ZRP-001234",
        "password": "s3cr3t"
    }

    Response  200
    -------------
    {
        "access":  "<short-lived JWT — send as Authorization: Bearer <access>>",
        "refresh": "<long-lived JWT — store securely, use to get new access tokens>",
        "user": {
            "id": 1,
            "username": "jdoe",
            "fullname": "John Doe",
            "zrp_badge_number": "ZRP-001234",
            "role": "analyst"
        }
    }
    """
    permission_classes    = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        zrp_badge_number = serializer.validated_data["zrp_badge_number"]
        password         = serializer.validated_data["password"]

        # authenticate() uses USERNAME_FIELD ('zrp_badge_number') internally
        user = authenticate(request, username=zrp_badge_number, password=password)

        if user is None:
            return Response(
                {"detail": "Invalid badge number or password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not user.is_active:
            return Response(
                {"detail": "This account has been disabled. Contact your administrator."},
                status=status.HTTP_403_FORBIDDEN,
            )

        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "access":  str(refresh.access_token),
                "refresh": str(refresh),
                "user": {
                    "id":               user.pk,
                    "username":         user.username,
                    "fullname":         user.fullname,
                    "zrp_badge_number": user.zrp_badge_number,
                    "role":             user.role,
                },
            },
            status=status.HTTP_200_OK,
        )


class TokenRefreshView(APIView):
    """
    POST /api/public/auth/token/refresh/

    Exchange a valid refresh token for a new access token.
    If ROTATE_REFRESH_TOKENS = True in settings, a new refresh token is also
    returned and the old one is blacklisted automatically by simplejwt.

    Request body
    ------------
    { "refresh": "<refresh JWT>" }

    Response  200
    -------------
    { "access": "<new access JWT>" }
    (+ "refresh": "<new refresh JWT>"  when rotation is enabled)
    """
    permission_classes    = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = TokenRefreshSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            refresh     = RefreshToken(serializer.validated_data["refresh"])
            access_token = str(refresh.access_token)

            # If simplejwt is configured to rotate refresh tokens, issue a new one
            from django.conf import settings
            simplejwt_settings = getattr(settings, "SIMPLE_JWT", {})
            rotate = simplejwt_settings.get("ROTATE_REFRESH_TOKENS", False)

            if rotate:
                # Blacklist the old token (requires token_blacklist app)
                try:
                    refresh.blacklist()
                except Exception:
                    pass  # blacklist app may not be installed
                new_refresh = RefreshToken.for_user(
                    CustomUser.objects.get(pk=refresh["user_id"])
                )
                return Response(
                    {"access": access_token, "refresh": str(new_refresh)},
                    status=status.HTTP_200_OK,
                )

            return Response({"access": access_token}, status=status.HTTP_200_OK)

        except TokenError as e:
            raise InvalidToken(e.args[0])


class LogoutView(APIView):
    """
    POST /api/public/auth/logout/

    Blacklist the provided refresh token so it can no longer be used to
    obtain new access tokens.  The current access token will still be
    valid until it expires (typical lifetime: 5–15 minutes).

    Requires 'rest_framework_simplejwt.token_blacklist' in INSTALLED_APPS
    and that the blacklist migration has been run.

    Request body
    ------------
    { "refresh": "<refresh JWT>" }

    Response  205  (Reset Content — tells the client to clear stored tokens)
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TokenRefreshSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            token = RefreshToken(serializer.validated_data["refresh"])
            token.blacklist()
            logger.info(f"User {request.user} logged out — refresh token blacklisted.")
            return Response(
                {"detail": "Successfully logged out."},
                status=status.HTTP_205_RESET_CONTENT,
            )
        except TokenError:
            return Response(
                {"detail": "Token is invalid or has already been blacklisted."},
                status=status.HTTP_400_BAD_REQUEST,
            )


# =============================================================================
# Public endpoints (Flutter mobile app)
# =============================================================================

class PublicCrimeMapView(APIView):
    """
    GET /api/public/crimes/
    Returns anonymized crime pins for the Flutter map.
    Supports ?crime_type_id=, ?suburb=, ?start_date=, ?end_date= filters.
    """
    permission_classes    = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = _filter_incidents(request).exclude(location__isnull=True)
        serializer = PublicCrimeIncidentSerializer(qs, many=True)
        return Response(serializer.data)


class PublicCrimeTypeListView(APIView):
    """GET /api/public/crime-types/ — list of crime categories + icons."""
    permission_classes    = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = CrimeType.objects.annotate(incident_count=Count("incidents"))
        serializer = CrimeTypeSerializer(qs, many=True)
        return Response(serializer.data)


# =============================================================================
# ZRP Dashboard — Incident CRUD
# =============================================================================

class IncidentListCreateView(APIView):
    """
    GET  /api/zrp/incidents/   — paginated, filtered incident list
    POST /api/zrp/incidents/   — create a new incident
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = _filter_incidents(request)
        serializer = CrimeIncidentSerializer(qs, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CrimeIncidentSerializer(
            data=request.data, context={"request": request}
        )
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class IncidentDetailView(APIView):
    """
    GET    /api/zrp/incidents/<id>/
    PUT    /api/zrp/incidents/<id>/
    DELETE /api/zrp/incidents/<id>/
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_object(self, pk):
        try:
            return CrimeIncident.objects.select_related("crime_type", "created_by").get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return None

    def get(self, request, pk):
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeIncidentSerializer(incident)
        return Response(serializer.data)

    def put(self, request, pk):
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeIncidentSerializer(
            incident, data=request.data, partial=True, context={"request": request}
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        permission_classes = [IsZRPAdmin]
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        incident.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class IncidentSimilarCasesView(APIView):
    """
    GET /api/zrp/incidents/<id>/similar/
    Returns the top-N most similar cases using the ProfileMatcher ML model.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def get(self, request, pk):
        try:
            incident = CrimeIncident.objects.get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        top_n   = int(request.query_params.get("top_n", 5))
        matcher = ProfileMatcher()
        try:
            similar_ids = matcher.find_similar(incident, top_n=top_n)
        except Exception as exc:
            logger.error(f"ProfileMatcher error: {exc}")
            return Response(
                {"detail": "Profile matching unavailable — model may not be trained yet."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        similar_qs = CrimeIncident.objects.filter(pk__in=similar_ids).select_related("crime_type")
        serializer = CrimeIncidentSerializer(similar_qs, many=True)
        return Response({"source_incident": pk, "similar_cases": serializer.data})


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
# ZRP Dashboard — KPI summary
# =============================================================================

class DashboardSummaryView(APIView):
    """GET /api/zrp/dashboard/summary/"""
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        now   = timezone.now()
        week  = now - timedelta(days=7)
        month = now - timedelta(days=30)

        total          = CrimeIncident.objects.count()
        last_7_days    = CrimeIncident.objects.filter(timestamp__gte=week).count()
        last_30_days   = CrimeIncident.objects.filter(timestamp__gte=month).count()
        by_status      = dict(
            CrimeIncident.objects.values_list("status")
                                 .annotate(c=Count("id"))
                                 .values_list("status", "c")
        )
        by_crime_type  = list(
            CrimeIncident.objects.values("crime_type__name")
                                 .annotate(count=Count("id"))
                                 .order_by("-count")[:10]
        )
        return Response(
            {
                "total_incidents": total,
                "last_7_days":     last_7_days,
                "last_30_days":    last_30_days,
                "by_status":       by_status,
                "top_crime_types": by_crime_type,
            }
        )


# =============================================================================
# Crime Type CRUD
# =============================================================================

class CrimeTypeListCreateView(APIView):
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = CrimeType.objects.annotate(incident_count=Count("incidents"))
        return Response(CrimeTypeSerializer(qs, many=True).data)

    def post(self, request):
        serializer = CrimeTypeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CrimeTypeDetailView(APIView):
    permission_classes = [IsZRPAuthenticated]

    def _get_object(self, pk):
        try:
            return CrimeType.objects.get(pk=pk)
        except CrimeType.DoesNotExist:
            return None

    def get(self, request, pk):
        obj = self._get_object(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeTypeSerializer(obj).data)

    def put(self, request, pk):
        obj = self._get_object(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeTypeSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        obj = self._get_object(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Analytics
# =============================================================================

class HeatmapView(APIView):
    """GET/POST /api/zrp/analytics/heatmap/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        serializer = HeatmapRequestSerializer(data=request.data or request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        d = serializer.validated_data

        qs = CrimeIncident.objects.exclude(location__isnull=True)
        if d.get("crime_type_id"):
            qs = qs.filter(crime_type_id=d["crime_type_id"])
        if d.get("start_date"):
            qs = qs.filter(timestamp__date__gte=d["start_date"])
        if d.get("end_date"):
            qs = qs.filter(timestamp__date__lte=d["end_date"])

        # Extract coordinates from PostGIS PointField
        coords = [
            (inc.location.y, inc.location.x)
            for inc in qs.only("location")
            if inc.location
        ]
        if not coords:
            return Response({"heatmap_data": []})

        result = compute_kde_heatmap(coords, bandwidth=d.get("bandwidth", 0.01))
        return Response({"heatmap_data": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class TimeSeriesView(APIView):
    """GET/POST /api/zrp/analytics/timeseries/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        serializer = TimeSeriesRequestSerializer(data=request.data or request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        d = serializer.validated_data

        qs = CrimeIncident.objects.all()
        if d.get("crime_type_id"):
            qs = qs.filter(crime_type_id=d["crime_type_id"])
        if d.get("start_date"):
            qs = qs.filter(timestamp__date__gte=d["start_date"])
        if d.get("end_date"):
            qs = qs.filter(timestamp__date__lte=d["end_date"])

        df = pd.DataFrame(list(qs.values("timestamp")))
        if df.empty:
            return Response({"timeseries": []})

        result = compute_time_series(df, freq=d.get("freq", "W"))
        return Response({"timeseries": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class HotspotView(APIView):
    """GET/POST /api/zrp/analytics/hotspots/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        qs = _filter_incidents(request).exclude(location__isnull=True)
        coords = [
            (inc.location.y, inc.location.x)
            for inc in qs.only("location")
            if inc.location
        ]
        if not coords:
            return Response({"hotspots": []})
        result = compute_hotspot_summary(coords)
        return Response({"hotspots": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class ProfileMatchView(APIView):
    """POST /api/zrp/analytics/profile-match/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        serializer = ProfileMatchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            incident = CrimeIncident.objects.get(pk=serializer.validated_data["incident_id"])
        except CrimeIncident.DoesNotExist:
            return Response({"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND)

        top_n   = serializer.validated_data.get("top_n", 5)
        matcher = ProfileMatcher()
        try:
            similar_ids = matcher.find_similar(incident, top_n=top_n)
        except Exception as exc:
            logger.error(f"ProfileMatcher error: {exc}")
            return Response(
                {"detail": "Profile matching unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        similar_qs = CrimeIncident.objects.filter(pk__in=similar_ids).select_related("crime_type")
        return Response(
            {
                "query_incident": serializer.validated_data["incident_id"],
                "matches":        CrimeIncidentSerializer(similar_qs, many=True).data,
            }
        )


# =============================================================================
# Admin — ML training trigger
# =============================================================================

class MLTrainView(APIView):
    """POST /api/zrp/ml/train/ — re-train the ProfileMatcher model."""
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        try:
            qs = CrimeIncident.objects.exclude(location__isnull=True)
            df = pd.DataFrame(
                list(
                    qs.values(
                        "id", "crime_type_id", "time_of_day", "day_of_week",
                        "weapon_used", "num_suspects", "modus_operandi",
                    )
                )
            )
            if df.empty:
                return Response(
                    {"detail": "No incident data available for training."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            matcher = ProfileMatcher()
            matcher.train(df)
            return Response(
                {"detail": f"Model trained successfully on {len(df)} incidents."},
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            logger.error(f"ML training failed: {exc}")
            return Response(
                {"detail": f"Training failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        


# =============================================================================
# Admin — User management
# =============================================================================

class UserListCreateView(APIView):
    permission_classes = [IsZRPAdmin]

    def get(self, request):
        qs = CustomUser.objects.all()
        return Response(UserSerializer(qs, many=True).data)

    def post(self, request):
        serializer = CreateUserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserDetailView(APIView):
    permission_classes = [IsZRPAdmin]

    def _get_object(self, pk):
        try:
            return CustomUser.objects.get(pk=pk)
        except CustomUser.DoesNotExist:
            return None

    def get(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(UserSerializer(user).data)

    def put(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        # Only allow updating specific safe fields
        allowed = {k: v for k, v in request.data.items() if k in ("role", "is_active", "base_station")}
        for field, value in allowed.items():
            setattr(user, field, value)
        user.save(update_fields=list(allowed.keys()))
        return Response(UserSerializer(user).data)

    def delete(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        user.is_active = False  # soft delete
        user.save(update_fields=["is_active"])
        return Response(status=status.HTTP_204_NO_CONTENT)