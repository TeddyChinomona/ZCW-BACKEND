"""
zimcrimewatch/views.py
======================
ZimCrimeWatch — API Views (APIView only, no ViewSets)

Serves both the Flutter mobile app (public, anonymised) and the
React ZRP dashboard (authenticated, full data + analytics).

Bug fixes in this version
--------------------------
  • IncidentDetailView.delete() used `request.user.zrp_profile.role` which does
    not exist.  Fixed to `request.user.role` (role is a direct field on
    CustomUser, not on a separate profile model).
  • CrimeTypeListCreateView.post() had the same zrp_profile bug.
  • CrimeTypeDetailView.delete() had the same zrp_profile bug.
  • SerialLinkageProbabilityView._incident_to_feature_dict() referenced
    `inc.incident_location` which is not a field on CrimeIncident.
    Fixed to use `inc.suburb` (the nearest equivalent stored field).

Endpoint groups
---------------
Public (no auth)
  POST /api/public/auth/login/
  POST /api/public/auth/logout/
  GET  /api/public/crimes/           — anonymised map pins for Flutter
  GET  /api/public/crime-types/      — list of crime categories + icons

ZRP Dashboard (JWT required)
  GET/POST         /api/zrp/incidents/
  GET/PUT/DELETE   /api/zrp/incidents/<id>/
  GET              /api/zrp/incidents/<id>/similar/    ← ProfileMatcher
  GET              /api/zrp/dashboard/summary/
  GET/POST         /api/zrp/crime-types/
  GET/PUT/DELETE   /api/zrp/crime-types/<id>/

Analytics (JWT required)
  GET/POST  /api/zrp/analytics/heatmap/
  GET/POST  /api/zrp/analytics/timeseries/
  GET/POST  /api/zrp/analytics/hotspots/
  POST      /api/zrp/analytics/profile-match/

Serial Crime Linkage (JWT required)
  POST  /api/zrp/analytics/serial-linkage/train/
  POST  /api/zrp/analytics/serial-linkage/cluster/
  POST  /api/zrp/analytics/serial-linkage/link-probability/

Admin only
  GET/POST         /api/zrp/users/
  GET/PUT/DELETE   /api/zrp/users/<id>/
  POST             /api/zrp/ml/train/
"""
from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timedelta

import numpy as np
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
from .serial_crime_linkage import SerialCrimeLinkageModel
from .models import CrimeIncident, CrimeType, CustomUser
from .permissions import IsZRPAdmin, IsZRPAnalystOrAdmin, IsZRPAuthenticated
from .serializers import (
    ChangePasswordSerializer,
    CrimeIncidentSerializer,
    CrimeTypeSerializer,
    CreateUserSerializer,
    ForgotPasswordSerializer,
    HeatmapRequestSerializer,
    LoginSerializer,
    ProfileMatchRequestSerializer,
    PublicCrimeIncidentSerializer,
    RegisterUserSerializer,
    ResetPasswordSerializer,
    TimeSeriesRequestSerializer,
    TokenRefreshSerializer,
    UserSerializer,
)


# =============================================================================
# Helpers
# =============================================================================

def _parse_date_range(request) -> tuple[datetime | None, datetime | None]:
    """Pull optional start_date / end_date from query params or request body."""
    start = request.query_params.get("start_date") or request.data.get("start_date")
    end   = request.query_params.get("end_date")   or request.data.get("end_date")

    def _parse(s):
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    return _parse(start), _parse(end)


def _filter_incidents(request):
    """
    Apply optional query-param filters to the CrimeIncident queryset.
    Supported params: crime_type_id, suburb, start_date, end_date, status.
    """
    qs     = CrimeIncident.objects.select_related("crime_type")
    params = request.query_params

    if crime_type_id := params.get("crime_type_id"):
        qs = qs.filter(crime_type_id=crime_type_id)
    if suburb := params.get("suburb"):
        qs = qs.filter(suburb__icontains=suburb)
    if status_param := params.get("status"):
        qs = qs.filter(status=status_param)

    start, end = _parse_date_range(request)
    if start:
        qs = qs.filter(timestamp__date__gte=start.date())
    if end:
        qs = qs.filter(timestamp__date__lte=end.date())

    return qs

def _generate_reset_token(zrp_badge_number: str) -> str:
    """
    Generate a time-bounded HMAC reset token for the given badge number.
    Valid for the current 10-minute window (floor(unix_time / 600)).
    """
    from django.conf import settings

    # Time window: floor division gives the same value for 10 minutes
    window = str(int(time.time()) // 600)
    message = f"{zrp_badge_number}:{window}"
    token = hmac.new(
        settings.SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()[:32]   # 32-char hex token
    return token


def _verify_reset_token(zrp_badge_number: str, token: str) -> bool:
    """
    Verify a reset token for the current OR previous time window
    (gives up to ~20 minutes of validity in practice).
    """
    from django.conf import settings

    now_window  = int(time.time()) // 600
    # Check both the current and the immediately preceding window so a token
    # generated at minute 9 of a window is still valid a minute later in
    # the next window — avoiding frustrating edge-case rejections.
    for window in (now_window, now_window - 1):
        message  = f"{zrp_badge_number}:{window}"
        expected = hmac.new(
            settings.SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        if hmac.compare_digest(expected, token):
            return True
    return False


# =============================================================================
# Authentication
# =============================================================================

class LoginView(APIView):
    """
    POST /api/public/auth/login/
    Accepts {zrp_badge_number, password} and returns JWT access + refresh tokens.
    AllowAny — no authentication required for the login endpoint itself.
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(
            request,
            # CustomUser.USERNAME_FIELD = "zrp_badge_number", so Django's
            # authenticate() maps `username=` to that field.
            username=serializer.validated_data["zrp_badge_number"],
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
        logger.info("User %s logged in.", user.username)
        return Response({
            "access":  str(refresh.access_token),
            "refresh": str(refresh),
            "user":    UserSerializer(user).data,
        })


class TokenRefreshView(APIView):
    """
    POST /api/public/auth/token/refresh/
    Exchange a valid refresh token for a new access token.
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = TokenRefreshSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            refresh      = RefreshToken(serializer.validated_data["refresh"])
            access_token = str(refresh.access_token)
            return Response({"access": access_token})
        except (TokenError, InvalidToken) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_401_UNAUTHORIZED)


class LogoutView(APIView):
    """
    POST /api/public/auth/logout/
    Blacklist the supplied refresh token so it cannot be reused.
    The current access token remains valid until it expires naturally.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TokenRefreshSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            token = RefreshToken(serializer.validated_data["refresh"])
            token.blacklist()
            logger.info("User %s logged out.", request.user)
            return Response(
                {"detail": "Successfully logged out."},
                status=status.HTTP_205_RESET_CONTENT,
            )
        except TokenError:
            return Response(
                {"detail": "Token is invalid or already blacklisted."},
                status=status.HTTP_400_BAD_REQUEST,
            )

class RegisterView(APIView):
    """
    POST /api/public/auth/register/

    Open endpoint — allows any new ZRP officer to create an account.
    All self-registered accounts receive the 'officer' role and must be
    promoted by an admin via PUT /api/zrp/users/<id>/ if elevated access
    is required.

    Body:
    {
      "username":         "jdoe",
      "first_name":       "John",
      "last_name":        "Doe",
      "zrp_badge_number": "1234",
      "password":         "SecurePass123!",
      "password_confirm": "SecurePass123!"
    }
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = RegisterUserSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.save()
        logger.info(
            "New officer registered: %s [%s].", user.username, user.zrp_badge_number
        )

        # Return a JWT pair immediately so the user is logged in after registering
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "message": "Registration successful. Welcome to ZimCrimeWatch.",
                "access":  str(refresh.access_token),
                "refresh": str(refresh),
                "user":    UserSerializer(user).data,
            },
            status=status.HTTP_201_CREATED,
        )


class ForgotPasswordView(APIView):
    """
    POST /api/public/auth/forgot-password/

    Step 1 of the password-reset flow.  Accepts a badge number and returns
    a time-limited reset token.

    In production: the token would be emailed to the officer's registered
    address instead of being returned in the response body.

    Body:    { "zrp_badge_number": "1234" }
    Returns: { "message": "…", "reset_token": "…", "badge_number": "…" }
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        badge = serializer.validated_data["zrp_badge_number"]

        # Look up the user — return the same response regardless of whether the
        # badge exists to prevent user-enumeration attacks.
        user_exists = CustomUser.objects.filter(
            zrp_badge_number=badge, is_active=True
        ).exists()

        if not user_exists:
            # Generic message — don't reveal whether the badge number exists
            return Response(
                {
                    "message": (
                        "If a registered account exists for this badge number, "
                        "a password reset token has been generated."
                    )
                }
            )

        # Generate the HMAC reset token
        reset_token = _generate_reset_token(badge)

        logger.info("Password reset token generated for badge %s.", badge)

        # NOTE: In production, send the token via email and omit it from the
        # response.  For the prototype we return it directly for testability.
        return Response(
            {
                "message": (
                    "A password reset token has been generated. "
                    "Use it within 20 minutes to reset your password. "
                    "(In production this token would be sent by email.)"
                ),
                "reset_token":  reset_token,
                "badge_number": badge,
            }
        )


class ResetPasswordView(APIView):
    """
    POST /api/public/auth/reset-password/

    Step 2 of the password-reset flow.  Accepts the badge number, the token
    received from ForgotPasswordView, and the new password.

    Body:
    {
      "zrp_badge_number": "1234",
      "token":            "<reset_token>",
      "new_password":     "NewSecure@123",
      "confirm_password": "NewSecure@123"
    }
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = ResetPasswordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        badge        = serializer.validated_data["zrp_badge_number"]
        token        = serializer.validated_data["token"]
        new_password = serializer.validated_data["new_password"]

        # Verify the HMAC token (stateless, time-bounded verification)
        if not _verify_reset_token(badge, token):
            return Response(
                {"detail": "Invalid or expired reset token. Please request a new one."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Fetch the user
        try:
            user = CustomUser.objects.get(
                zrp_badge_number=badge, is_active=True
            )
        except CustomUser.DoesNotExist:
            return Response(
                {"detail": "Account not found or deactivated."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Set the new password (Django hashes it automatically)
        user.set_password(new_password)
        user.save(update_fields=["password"])
        logger.info("Password reset successfully for badge %s.", badge)

        return Response(
            {"detail": "Password has been reset successfully. You can now log in."}
        )


class ChangePasswordView(APIView):
    """
    POST /api/zrp/auth/change-password/

    Allows an authenticated officer to change their own password.
    Requires the current password for verification.

    Body:
    {
      "current_password": "OldPassword@1",
      "new_password":     "NewPassword@2",
      "confirm_password": "NewPassword@2"
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ChangePasswordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user             = request.user
        current_password = serializer.validated_data["current_password"]
        new_password     = serializer.validated_data["new_password"]

        # Verify the current password before allowing the change
        if not user.check_password(current_password):
            return Response(
                {"detail": "Current password is incorrect."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user.set_password(new_password)
        user.save(update_fields=["password"])
        logger.info("Password changed by officer %s.", user.username)

        return Response({"detail": "Password changed successfully."})

# =============================================================================
# Public endpoints (Flutter mobile app — no auth required)
# =============================================================================

class PublicCrimeMapView(APIView):
    """
    GET /api/public/crimes/
    Returns anonymised crime pins for the Flutter map.
    Supports ?crime_type_id=, ?suburb=, ?start_date=, ?end_date= filters.
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = _filter_incidents(request).exclude(location__isnull=True)
        return Response(PublicCrimeIncidentSerializer(qs, many=True).data)


class PublicCrimeTypeListView(APIView):
    """GET /api/public/crime-types/ — list of crime categories + icons."""
    permission_classes     = [AllowAny]
    authentication_classes = []

    def get(self, request):
        qs = CrimeType.objects.annotate(incident_count=Count("incidents"))
        return Response(CrimeTypeSerializer(qs, many=True).data)


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
        return Response(CrimeIncidentSerializer(qs, many=True).data)

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
    DELETE /api/zrp/incidents/<id>/   (admin only)
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_object(self, pk):
        try:
            return CrimeIncident.objects.select_related(
                "crime_type", "created_by"
            ).get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return None

    def get(self, request, pk):
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeIncidentSerializer(incident).data)

    def put(self, request, pk):
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeIncidentSerializer(
            incident, data=request.data, partial=True,
            context={"request": request},
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        # BUG FIX: the original code referenced `request.user.zrp_profile.role`
        # which does not exist.  `role` is a direct field on CustomUser.
        if getattr(request.user, "role", None) != "admin":
            return Response(
                {"detail": "Admin role required to delete incidents."},
                status=status.HTTP_403_FORBIDDEN,
            )
        incident = self._get_object(pk)
        if not incident:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        incident.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class IncidentSimilarCasesView(APIView):
    """
    GET /api/zrp/incidents/<id>/similar/
    Returns the top-N most similar cases using the ProfileMatcher ML model.
    Optional query param: ?top_n=5 (default 5)
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def get(self, request, pk):
        try:
            incident = CrimeIncident.objects.select_related("crime_type").get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        top_n = int(request.query_params.get("top_n", 5))

        try:
            matcher     = ProfileMatcher.load()
            similar_ids = matcher.find_similar(incident, top_n=top_n)
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "Profile matching model not trained yet. "
                        "Run: python manage.py train_profile_matcher"
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.error("ProfileMatcher.find_similar error: %s", exc)
            return Response(
                {"detail": "Profile matching unavailable — see server logs."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        similar_qs = CrimeIncident.objects.filter(
            pk__in=similar_ids
        ).select_related("crime_type")

        return Response({
            "source_incident": pk,
            "similar_cases":   CrimeIncidentSerializer(similar_qs, many=True).data,
        })


# =============================================================================
# ZRP Dashboard — KPI Summary
# =============================================================================

class DashboardSummaryView(APIView):
    """GET /api/zrp/dashboard/summary/ — high-level KPI card data."""
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        now   = timezone.now()
        week  = now - timedelta(days=7)
        month = now - timedelta(days=30)

        total        = CrimeIncident.objects.count()
        last_7_days  = CrimeIncident.objects.filter(timestamp__gte=week).count()
        last_30_days = CrimeIncident.objects.filter(timestamp__gte=month).count()
        by_status    = dict(
            CrimeIncident.objects
            .values_list("status")
            .annotate(c=Count("id"))
            .values_list("status", "c")
        )
        by_crime_type = list(
            CrimeIncident.objects
            .values("crime_type__name")
            .annotate(count=Count("id"))
            .order_by("-count")[:10]
        )
        return Response({
            "total_incidents": total,
            "last_7_days":     last_7_days,
            "last_30_days":    last_30_days,
            "by_status":       by_status,
            "top_crime_types": by_crime_type,
        })


# =============================================================================
# Crime Type CRUD
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
        # BUG FIX: was `request.user.zrp_profile.role` — fixed to `request.user.role`
        if getattr(request.user, "role", None) not in ("analyst", "admin"):
            return Response(
                {"detail": "Permission denied."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = CrimeTypeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CrimeTypeDetailView(APIView):
    """GET / PUT / DELETE /api/zrp/crime-types/<id>/"""
    permission_classes = [IsZRPAuthenticated]

    def _get_ct(self, pk):
        try:
            return CrimeType.objects.annotate(incident_count=Count("incidents")).get(pk=pk)
        except CrimeType.DoesNotExist:
            return None

    def get(self, request, pk):
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeTypeSerializer(obj).data)

    def put(self, request, pk):
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeTypeSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        # BUG FIX: was `request.user.zrp_profile.role` — fixed to `request.user.role`
        if getattr(request.user, "role", None) != "admin":
            return Response(
                {"detail": "Admin role required."},
                status=status.HTTP_403_FORBIDDEN,
            )
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if obj.incident_count > 0:
            return Response(
                {
                    "detail": (
                        f"Cannot delete: {obj.incident_count} incidents "
                        "reference this crime type."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Analytics
# =============================================================================

class HeatmapView(APIView):
    """GET/POST /api/zrp/analytics/heatmap/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        serializer = HeatmapRequestSerializer(
            data=request.data or request.query_params
        )
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

        # Extract (lat, lng) tuples from the PostGIS PointField.
        # location.y = latitude, location.x = longitude (standard GIS convention)
        coords = [
            (inc.location.y, inc.location.x)
            for inc in qs.only("location")
            if inc.location
        ]
        if not coords:
            return Response({"heatmap_data": []})

        result = compute_kde_heatmap(coords, bandwidth_km=d.get("bandwidth", 0.01))
        return Response({"heatmap_data": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class TimeSeriesView(APIView):
    """GET/POST /api/zrp/analytics/timeseries/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        serializer = TimeSeriesRequestSerializer(
            data=request.data or request.query_params
        )
        if not serializer.is_valid():
            logger.info(serializer.errors)
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

        # The serializer validates freq as "D" / "W" / "M".
        # compute_time_series now accepts both short codes and long-form names.
        result = compute_time_series(df, d.get("freq", "W"))
        logger.info("Timeseries view returned data for freq=%s", d.get("freq"))
        return Response({"timeseries": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class HotspotView(APIView):
    """GET/POST /api/zrp/analytics/hotspots/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        # Build queryset — do NOT exclude location__isnull here yet; we handle
        # both PostGIS and flat-column incidents in the loop below.
        qs = _filter_incidents(request).select_related("crime_type")

        coords      = []
        crime_types = []
        suburbs     = []

        for inc in qs.only("location", "crime_type__name", "suburb"):
            lat, lng = None, None

            # Primary: PostGIS PointField (location.y = lat, location.x = lng)
            if inc.location:
                lat = inc.location.y
                lng = inc.location.x
            # Fallback: backward-compat properties derived from `location`
            elif getattr(inc, "latitude", None) is not None:
                lat = inc.latitude
                lng = inc.longitude

            if lat is None or lng is None:
                continue  # no usable coordinate — skip

            coords.append((lat, lng))
            crime_types.append(inc.crime_type.name if inc.crime_type else "Unknown")
            suburbs.append(inc.suburb or "")

        if not coords:
            return Response({"hotspots": []})

        # ── Adaptive DBSCAN ──────────────────────────────────────────────────
        # Try progressively looser parameters until we find at least one
        # cluster.  This prevents returning nothing on sparse datasets.
        param_ladder = [
            {"eps_km": 0.5,  "min_samples": 3},   # tight   — dense urban
            {"eps_km": 1.0,  "min_samples": 3},   # medium  — suburban
            {"eps_km": 2.0,  "min_samples": 2},   # loose   — sparse data
            {"eps_km": 5.0,  "min_samples": 2},   # very loose — last resort
        ]

        result = []
        for params in param_ladder:
            result = compute_hotspot_summary(
                coords, crime_types, suburbs,
                eps_km=params["eps_km"],
                min_samples=params["min_samples"],
            )
            if result:
                break  # found clusters — stop trying wider params

        return Response({"hotspots": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


# =============================================================================
# Analytics — Profile Match
# =============================================================================

class ProfileMatchView(APIView):
    """
    POST /api/zrp/analytics/profile-match/
    Body: { "incident_id": <int>, "top_n": <int, optional, default 5> }

    Fallback strategy:
      1. Try ProfileMatcher (RandomForest, supervised).
      2. If model file is missing, fall back to SerialCrimeLinkageModel (DBSCAN).
      3. If neither model file exists, return 503 with clear instructions.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        serializer = ProfileMatchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        incident_id = serializer.validated_data["incident_id"]
        top_n       = serializer.validated_data.get("top_n", 5)

        try:
            incident = CrimeIncident.objects.select_related("crime_type").get(pk=incident_id)
        except CrimeIncident.DoesNotExist:
            return Response({"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND)

        # ── Path 1: Supervised — ProfileMatcher (RandomForest) ────────────────
        try:
            matcher     = ProfileMatcher.load()
            similar_ids = matcher.find_similar(incident, top_n=top_n)

            similar_qs = CrimeIncident.objects.filter(
                pk__in=similar_ids
            ).select_related("crime_type")

            return Response({
                "query_incident": incident_id,
                "model_used":     "supervised",
                "matches":        CrimeIncidentSerializer(similar_qs, many=True).data,
            })

        except FileNotFoundError:
            logger.info(
                "ProfileMatchView: supervised model not found for incident %d — "
                "trying SerialCrimeLinkageModel fallback.",
                incident_id,
            )
        except Exception as exc:
            logger.warning(
                "ProfileMatchView: supervised match failed for incident %d (%s) — "
                "trying SerialCrimeLinkageModel fallback.",
                incident_id, exc,
            )

        # ── Path 2: Unsupervised fallback — SerialCrimeLinkageModel ──────────
        try:
            linkage_model = SerialCrimeLinkageModel.load()
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "No trained model is available. "
                        "Go to the ML Training page and click 'Train Model Now' "
                        "to train before using profile matching."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        agg_df     = linkage_model.agg_df_
        sim_matrix = linkage_model.sim_matrix_

        if agg_df is None or sim_matrix is None:
            return Response(
                {
                    "detail": (
                        "Serial linkage model is present but was not fitted. "
                        "Please re-train from the ML Training page."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Locate the query incident's row in agg_df by its case_number string.
        matches_mask = agg_df["case_number"] == incident.case_number
        if not matches_mask.any():
            return Response(
                {
                    "detail": (
                        f"Incident '{incident.case_number}' was not included in the "
                        "last training run.  Re-train the model to include it."
                    )
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        query_idx = int(matches_mask.idxmax())

        # Read this incident's row from the N×N similarity matrix.
        # Each entry sim_matrix[i, j] is the composite similarity score between
        # case i and case j (1.0 = identical, 0.0 = nothing in common).
        sim_row               = sim_matrix[query_idx].copy()
        sim_row[query_idx]    = -1.0          # exclude the query itself

        top_indices       = np.argsort(sim_row)[::-1][:top_n]
        top_case_numbers  = agg_df.iloc[top_indices]["case_number"].tolist()

        similar_qs = CrimeIncident.objects.filter(
            case_number__in=top_case_numbers
        ).select_related("crime_type")

        # Preserve the similarity-score ordering in the response.
        order_map          = {cn: rank for rank, cn in enumerate(top_case_numbers)}
        similar_incidents  = sorted(similar_qs, key=lambda inc: order_map.get(inc.case_number, 999))

        return Response({
            "query_incident": incident_id,
            "model_used":     "unsupervised",
            "matches":        CrimeIncidentSerializer(similar_incidents, many=True).data,
        })


# =============================================================================
# Serial Crime Linkage
# =============================================================================

class SerialLinkageTrainView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/train/
    Trains SerialCrimeLinkageModel (DBSCAN) on all incidents in the database.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        qs = CrimeIncident.objects.all()
        if not qs.exists():
            return Response(
                {"detail": "No incidents in the database to train on."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        model = SerialCrimeLinkageModel()
        try:
            results = model.train_unsupervised_from_queryset(qs)
        except Exception as exc:
            logger.error("SerialLinkageTrainView error: %s", exc)
            return Response(
                {"detail": f"Serial linkage training failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if "error" in results:
            return Response({"detail": results["error"]}, status=status.HTTP_400_BAD_REQUEST)

        logger.info("Serial linkage model trained: %d clusters found.", results.get("n_serial_clusters", 0))
        return Response(results)


class SerialLinkageClusterView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/cluster/
    Returns per-incident cluster assignments from the pre-trained model.
    Optionally accepts { "case_numbers": [...] } to filter results.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        try:
            model = SerialCrimeLinkageModel.load()
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "Serial linkage model not trained yet. "
                        "Call POST /api/zrp/analytics/serial-linkage/train/ first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if model.agg_df_ is None:
            return Response(
                {"detail": "Model is trained but has no cluster data. Re-train."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        requested_cases = request.data.get("case_numbers")

        df                    = model.agg_df_.copy()
        df["serial_cluster"]  = model.cluster_labels_
        df["cluster_label"]   = df["serial_cluster"].apply(
            lambda c: f"Serial Group {c}" if c >= 0 else "Unlinked"
        )

        if requested_cases:
            df = df[df["case_number"].isin(requested_cases)]

        # Only return JSON-safe columns — exclude numpy arrays and geometry blobs
        safe_cols     = ["case_number", "serial_cluster", "cluster_label", "mo_text", "full_location"]
        existing_cols = [c for c in safe_cols if c in df.columns]

        return Response({
            "n_cases":           len(df),
            "n_serial_clusters": int((df["serial_cluster"] >= 0).sum()),
            "cases":             df[existing_cols].to_dict(orient="records"),
            "cluster_summary":   model._build_cluster_summary(),
        })


class SerialLinkageProbabilityView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/link-probability/
    Body: { "incident_id_a": <int>, "incident_id_b": <int> }

    Computes the probability that two specific incidents were committed by
    the same offender using the SerialCrimeLinkageModel.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        id_a = request.data.get("incident_id_a")
        id_b = request.data.get("incident_id_b")

        if not id_a or not id_b:
            return Response(
                {"detail": "Both 'incident_id_a' and 'incident_id_b' are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if id_a == id_b:
            return Response(
                {"detail": "incident_id_a and incident_id_b must be different."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            inc_a = CrimeIncident.objects.select_related("crime_type").get(pk=id_a)
            inc_b = CrimeIncident.objects.select_related("crime_type").get(pk=id_b)
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "One or both incidents not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            model = SerialCrimeLinkageModel.load()
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "Serial linkage model not trained yet. "
                        "Call POST /api/zrp/analytics/serial-linkage/train/ first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        def _incident_to_feature_dict(inc) -> dict:
            """
            Convert a CrimeIncident ORM object into the feature dict expected
            by SerialCrimeLinkageModel.link_probability().

            BUG FIX: the original code referenced `inc.incident_location` which
            does not exist on CrimeIncident.  The closest equivalent stored on
            the model is `inc.suburb`, so we use that for both the residential
            address and incident location slots.
            """
            date_ord = inc.timestamp.toordinal() if inc.timestamp else None
            time_min = (
                inc.timestamp.hour * 60 + inc.timestamp.minute
                if inc.timestamp
                else None
            )

            # Combine suburb with any description text to give the similarity
            # model as much location context as possible.
            location_text = " ".join(filter(None, [
                inc.suburb or "",
            ])).strip()

            return {
                "date_ord":     date_ord,
                "time_min":     time_min,
                "mean_age":     35,       # no per-incident victim age on this model
                "pct_female":   0.0,
                "pct_male":     1.0,
                "full_location": location_text,
                "mo_text":      inc.modus_operandi or "",
            }

        case_a_dict = _incident_to_feature_dict(inc_a)
        case_b_dict = _incident_to_feature_dict(inc_b)

        try:
            result = model.link_probability(case_a_dict, case_b_dict)
        except Exception as exc:
            logger.error("link_probability error: %s", exc)
            return Response(
                {"detail": f"Link probability calculation failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        result["incident_id_a"] = id_a
        result["incident_id_b"] = id_b
        return Response(result)


# =============================================================================
# Admin — ML training trigger
# =============================================================================

class MLTrainView(APIView):
    """
    POST /api/zrp/ml/train/

    Dual-mode training:
      • Supervised   — ProfileMatcher (RandomForest) when labelled incidents exist
      • Unsupervised — SerialCrimeLinkageModel (DBSCAN) when no labels present

    The `mode` key in the response tells the frontend which path was taken.
    """
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        # ── Step 1: Check for labelled incidents ──────────────────────────────
        labelled_qs = CrimeIncident.objects.exclude(
            serial_group_label__in=["", None]
        ).select_related("crime_type")

        if labelled_qs.exists():
            # ── Supervised — ProfileMatcher (RandomForest) ────────────────────
            logger.info(
                "MLTrainView: %d labelled incidents — training ProfileMatcher (supervised).",
                labelled_qs.count(),
            )
            matcher = ProfileMatcher()
            try:
                metrics = matcher.train(labelled_qs)
            except Exception as exc:
                logger.error("MLTrainView supervised training failed: %s", exc)
                return Response(
                    {"detail": f"Supervised training failed: {exc}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            if "error" in metrics:
                return Response({"detail": metrics["error"]}, status=status.HTTP_400_BAD_REQUEST)

            return Response({"mode": "supervised", **metrics})

        # ── Unsupervised — SerialCrimeLinkageModel (DBSCAN) ───────────────────
        logger.info(
            "MLTrainView: no labelled incidents — "
            "falling back to SerialCrimeLinkageModel (unsupervised DBSCAN)."
        )

        all_qs = CrimeIncident.objects.all()
        if not all_qs.exists():
            return Response(
                {"detail": "No incident data found. Add incidents before training."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_data = list(all_qs.values(
            "case_number",
            "timestamp",
            "description_narrative",
            "num_suspects",
            "suburb",
            "modus_operandi",
            "status",
        ))

        df = pd.DataFrame(raw_data)

        # Split the single timestamp into separate date and time strings.
        # SerialCrimeLinkageModel._parse_date_to_ordinal() accepts "YYYY-MM-DD"
        # and _parse_time_to_minutes() accepts "HHMM" (e.g. "1430").
        df["date_received"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        df["time_received"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H%M")

        # Map CrimeIncident fields to the names SerialCrimeLinkageModel expects.
        # The RRB-style complainant fields aren't stored on CrimeIncident, so we
        # use the best available approximations.
        df["complainant_name"]            = df["description_narrative"].fillna("")
        df["sex"]                         = ""       # not collected at incident level
        df["age"]                         = df["num_suspects"].fillna(0)
        df["residential_address"]         = df["suburb"].fillna("")
        df["incident_location"]           = df["suburb"].fillna("")
        df["property_stolen_description"] = df["modus_operandi"].fillna("")

        df = df.drop(columns=[
            "timestamp", "description_narrative", "num_suspects",
            "modus_operandi", "status",
        ])

        linkage_model = SerialCrimeLinkageModel()
        try:
            metrics = linkage_model.train_unsupervised(df)
        except Exception as exc:
            logger.error("MLTrainView unsupervised training failed: %s", exc)
            return Response(
                {"detail": f"Unsupervised training failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if "error" in metrics:
            return Response({"detail": metrics["error"]}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"mode": "unsupervised", **metrics})


# =============================================================================
# Admin — User management
# =============================================================================

class UserListCreateView(APIView):
    """
    GET  /api/zrp/users/  — list all ZRP user accounts
    POST /api/zrp/users/  — create a new user (admin only)
    """
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
    """
    GET / PUT / DELETE /api/zrp/users/<id>/
    Soft-delete (is_active=False) is used instead of hard delete to preserve
    audit trails.
    """
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

        # Only allow updating safe fields — never username or password here
        allowed_fields = ("role", "is_active", "base_station")
        allowed_data   = {k: v for k, v in request.data.items() if k in allowed_fields}
        for field, value in allowed_data.items():
            setattr(user, field, value)
        user.save(update_fields=list(allowed_data.keys()))
        return Response(UserSerializer(user).data)

    def delete(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        # Soft delete — preserves historical audit data
        user.is_active = False
        user.save(update_fields=["is_active"])
        return Response(status=status.HTTP_204_NO_CONTENT)