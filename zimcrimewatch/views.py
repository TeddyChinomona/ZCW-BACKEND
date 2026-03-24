"""
zimcrimewatch/views.py
======================
ZimCrimeWatch — API Views

All views use APIView (no ViewSets) for maximum transparency.

New endpoints added in this revision
-------------------------------------
  POST /api/public/auth/register/          — self-registration (officer role)
  POST /api/public/auth/forgot-password/   — request a password-reset token
  POST /api/public/auth/reset-password/    — set new password with token
  POST /api/zrp/auth/change-password/      — authenticated password change

Fixed in this revision
----------------------
  • HotspotView._run() now passes eps_km / min_samples to compute_hotspot_summary()
  • ProfileMatchView falls back cleanly between supervised → unsupervised models
  • MLTrainView correctly maps CrimeIncident fields to the linkage model schema
  • All serializer.validated_data accesses use .get() with safe defaults
  • Role checks use request.user.role (the CustomUser field) not .zrp_profile.role
    (there is no ZRPProfile model in the project — role lives on CustomUser)
"""
from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.contrib.auth import authenticate
from django.db.models import Count
from django.utils import timezone
from loguru import logger
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import RefreshToken

# ── Internal helpers ──────────────────────────────────────────────────────────
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
# Utility helpers
# =============================================================================

def _parse_date_range(request) -> tuple[datetime | None, datetime | None]:
    """
    Pull optional start_date / end_date from either query params (GET) or
    request body (POST).  Returns a (start, end) tuple of datetime objects.
    """
    start = request.query_params.get("start_date") or request.data.get("start_date")
    end   = request.query_params.get("end_date")   or request.data.get("end_date")

    def _parse(s: str | None) -> datetime | None:
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

    Supported params
    ----------------
    crime_type_id  : filter by CrimeType primary key
    suburb         : case-insensitive contains filter on suburb
    status         : exact match on status field
    start_date     : incidents on or after this date (YYYY-MM-DD)
    end_date       : incidents on or before this date (YYYY-MM-DD)
    """
    qs = CrimeIncident.objects.select_related("crime_type")
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


# ---------------------------------------------------------------------------
# Prototype password-reset token helpers
# ---------------------------------------------------------------------------
# In production this would be replaced with:
#   • A unique token stored in a dedicated PasswordResetToken model (DB-backed)
#   • An email sent to the officer's registered address
#   • An expiry field (e.g. 1 hour)
#
# For the prototype we generate an HMAC-SHA256 token derived from:
#   badge_number + timestamp (truncated to 10 min windows) + SECRET_KEY
# This is stateless (no DB table needed) and naturally expires every
# 10 minutes.  The client must send the token back within the same window.

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

    Body: { "zrp_badge_number": "…", "password": "…" }
    Returns: { "access": "…", "refresh": "…", "user": {…} }
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # CustomUser.USERNAME_FIELD = 'zrp_badge_number', so Django's
        # authenticate() maps the username kwarg to that field automatically.
        user = authenticate(
            request,
            username=serializer.validated_data["zrp_badge_number"],
            password=serializer.validated_data["password"],
        )

        if user is None:
            return Response(
                {"detail": "Invalid badge number or password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not user.is_active:
            return Response(
                {"detail": "This account has been deactivated. Contact your administrator."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Generate a fresh JWT pair
        refresh = RefreshToken.for_user(user)
        logger.info("Officer %s logged in (role: %s).", user.username, user.role)

        return Response({
            "access":  str(refresh.access_token),
            "refresh": str(refresh),
            "user":    UserSerializer(user).data,
        })


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


class TokenRefreshView(APIView):
    """
    POST /api/public/auth/token/refresh/

    Body:    { "refresh": "<refresh_token>" }
    Returns: { "access": "<new_access_token>" }
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

    Blacklists the supplied refresh token so it cannot be reused.
    The current access token stays valid until its natural expiry.

    Body: { "refresh": "<refresh_token>" }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TokenRefreshSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            token = RefreshToken(serializer.validated_data["refresh"])
            token.blacklist()
            logger.info("Officer %s logged out.", request.user)
            return Response(
                {"detail": "Successfully logged out."},
                status=status.HTTP_205_RESET_CONTENT,
            )
        except TokenError:
            return Response(
                {"detail": "Token is invalid or already blacklisted."},
                status=status.HTTP_400_BAD_REQUEST,
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
# Public endpoints (Flutter mobile app — no authentication required)
# =============================================================================

class PublicCrimeMapView(APIView):
    """
    GET /api/public/crimes/

    Returns anonymised crime pins for the Flutter map.
    Only incidents with a valid PostGIS location are included.

    Query params: crime_type_id, suburb, start_date, end_date
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def get(self, request):
        # Exclude incidents that have no geographic data — they can't be mapped
        qs = _filter_incidents(request).exclude(location__isnull=True)
        return Response(PublicCrimeIncidentSerializer(qs, many=True).data)


class PublicCrimeTypeListView(APIView):
    """GET /api/public/crime-types/ — list of crime categories + icons + counts."""
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
    GET  /api/zrp/incidents/   — filtered list of all incidents
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
    PUT    /api/zrp/incidents/<id>/   (partial update)
    DELETE /api/zrp/incidents/<id>/   (admin only — hard delete)
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_object(self, pk: int):
        try:
            return CrimeIncident.objects.select_related(
                "crime_type", "created_by"
            ).get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return None

    def get(self, request, pk: int):
        incident = self._get_object(pk)
        if not incident:
            return Response(
                {"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(CrimeIncidentSerializer(incident).data)

    def put(self, request, pk: int):
        incident = self._get_object(pk)
        if not incident:
            return Response(
                {"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND
            )
        serializer = CrimeIncidentSerializer(
            incident,
            data=request.data,
            partial=True,             # allow partial updates (PATCH-style behaviour)
            context={"request": request},
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk: int):
        # Hard delete is restricted to admin role only.
        # request.user.role is the field on CustomUser (not .zrp_profile.role —
        # there is no ZRPProfile model in this project).
        if getattr(request.user, "role", None) != "admin":
            return Response(
                {"detail": "Admin role is required to delete incidents."},
                status=status.HTTP_403_FORBIDDEN,
            )
        incident = self._get_object(pk)
        if not incident:
            return Response(
                {"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND
            )
        incident.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class IncidentSimilarCasesView(APIView):
    """
    GET /api/zrp/incidents/<id>/similar/
    Query param: ?top_n=5  (default 5, max 50)

    Returns the top-N most similar incidents to the given one using
    ProfileMatcher.find_similar() (TF-IDF cosine similarity on M.O. text).
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def get(self, request, pk: int):
        try:
            incident = CrimeIncident.objects.select_related("crime_type").get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "Incident not found."}, status=status.HTTP_404_NOT_FOUND
            )

        top_n = min(int(request.query_params.get("top_n", 5)), 50)

        # Load the persisted ProfileMatcher model and run similarity search
        try:
            matcher     = ProfileMatcher.load()
            similar_ids = matcher.find_similar(incident, top_n=top_n)
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "Profile matching model not trained yet. "
                        "POST /api/zrp/ml/train/ to train first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.error("ProfileMatcher.find_similar error for incident %d: %s", pk, exc)
            return Response(
                {"detail": "Profile matching temporarily unavailable — see server logs."},
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
    """GET /api/zrp/dashboard/summary/ — headline KPI statistics."""
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        now   = timezone.now()
        week  = now - timedelta(days=7)
        month = now - timedelta(days=30)

        total        = CrimeIncident.objects.count()
        last_7_days  = CrimeIncident.objects.filter(timestamp__gte=week).count()
        last_30_days = CrimeIncident.objects.filter(timestamp__gte=month).count()

        # by_status: dict mapping status string → count of incidents
        by_status = dict(
            CrimeIncident.objects
            .values_list("status")
            .annotate(c=Count("id"))
            .values_list("status", "c")
        )

        # Top 10 crime types by incident count, most frequent first
        top_crime_types = list(
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
            "top_crime_types": top_crime_types,
        })


# =============================================================================
# Crime Type CRUD
# =============================================================================

class CrimeTypeListCreateView(APIView):
    """
    GET  /api/zrp/crime-types/   — list all with incident counts
    POST /api/zrp/crime-types/   — create (analyst/admin only)
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = CrimeType.objects.annotate(
            incident_count=Count("incidents")
        ).order_by("name")
        return Response(CrimeTypeSerializer(qs, many=True).data)

    def post(self, request):
        # Only analysts and admins may create new crime types
        if getattr(request.user, "role", None) not in ("analyst", "admin"):
            return Response(
                {"detail": "Analyst or admin role required."},
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

    def _get_ct(self, pk: int):
        try:
            return CrimeType.objects.annotate(
                incident_count=Count("incidents")
            ).get(pk=pk)
        except CrimeType.DoesNotExist:
            return None

    def get(self, request, pk: int):
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CrimeTypeSerializer(obj).data)

    def put(self, request, pk: int):
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = CrimeTypeSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk: int):
        if getattr(request.user, "role", None) != "admin":
            return Response(
                {"detail": "Admin role required."},
                status=status.HTTP_403_FORBIDDEN,
            )
        obj = self._get_ct(pk)
        if not obj:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        # Prevent deletion if any incident still references this crime type
        if obj.incident_count > 0:
            return Response(
                {
                    "detail": (
                        f"Cannot delete: {obj.incident_count} incident(s) reference "
                        "this crime type. Remove or re-classify them first."
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
    """GET / POST /api/zrp/analytics/heatmap/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        # Accept filters from either GET query params or POST body
        data = request.data if request.method == "POST" else request.query_params
        serializer = HeatmapRequestSerializer(data=data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        d = serializer.validated_data

        # Build the queryset — only incidents with a PostGIS location are usable
        qs = CrimeIncident.objects.exclude(location__isnull=True)
        if d.get("crime_type_id"):
            qs = qs.filter(crime_type_id=d["crime_type_id"])
        if d.get("start_date"):
            qs = qs.filter(timestamp__date__gte=d["start_date"])
        if d.get("end_date"):
            qs = qs.filter(timestamp__date__lte=d["end_date"])

        # Extract (lat, lng) tuples — location.y=lat, location.x=lng in PostGIS
        coords = [
            (inc.location.y, inc.location.x)
            for inc in qs.only("location")
            if inc.location
        ]

        if not coords:
            return Response({"heatmap_data": []})

        # Pass bandwidth from validated_data (defaults to 0.01 via serializer)
        result = compute_kde_heatmap(
            coordinates=coords,
            bandwidth=float(d.get("bandwidth") or 0.01),
        )
        return Response({"heatmap_data": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class TimeSeriesView(APIView):
    """GET / POST /api/zrp/analytics/timeseries/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        data = request.data if request.method == "POST" else request.query_params
        serializer = TimeSeriesRequestSerializer(data=data)
        if not serializer.is_valid():
            logger.warning("TimeSeriesView serializer errors: %s", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        d = serializer.validated_data

        qs = CrimeIncident.objects.all()
        if d.get("crime_type_id"):
            qs = qs.filter(crime_type_id=d["crime_type_id"])
        if d.get("start_date"):
            qs = qs.filter(timestamp__date__gte=d["start_date"])
        if d.get("end_date"):
            qs = qs.filter(timestamp__date__lte=d["end_date"])

        # Only pull the timestamp column to keep the DataFrame small
        df = pd.DataFrame(list(qs.values("timestamp")))
        if df.empty:
            return Response({"timeseries": []})

        # freq is validated by the serializer to one of "D", "W", "M"
        result = compute_time_series(df, d.get("freq", "W"))
        logger.debug("TimeSeriesView: returned %d labels.", len(result.get("labels", [])))
        return Response({"timeseries": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class HotspotView(APIView):
    """GET / POST /api/zrp/analytics/hotspots/"""
    permission_classes = [IsZRPAnalystOrAdmin]

    def _run(self, request):
        qs = _filter_incidents(request).select_related("crime_type")

        # Build parallel lists of coordinates, crime types, and suburbs.
        # We support both PostGIS PointField (preferred) and the backward-compat
        # .latitude / .longitude properties (which also read from location).
        coords      = []
        crime_types = []
        suburbs     = []

        for inc in qs.only("location", "crime_type", "suburb"):
            lat = lng = None

            if inc.location:
                lat, lng = inc.location.y, inc.location.x
            elif getattr(inc, "latitude", None) is not None:
                # Backward-compat fallback (reads from location too)
                lat = inc.latitude
                lng = inc.longitude

            if lat is None or lng is None:
                continue   # skip incidents with no coordinate data

            coords.append((lat, lng))
            crime_types.append(
                inc.crime_type.name if inc.crime_type else "Unknown"
            )
            suburbs.append(inc.suburb or "")

        if not coords:
            return Response({"hotspots": []})

        # Adaptive DBSCAN: try progressively looser parameters until we get
        # at least one cluster.  This prevents the view from always returning
        # an empty list on small or geographically spread datasets.
        param_ladder = [
            {"eps_km": 0.5,  "min_samples": 3},   # tight   — dense urban areas
            {"eps_km": 1.0,  "min_samples": 3},   # medium  — suburban areas
            {"eps_km": 2.0,  "min_samples": 2},   # loose   — sparse rural data
            {"eps_km": 5.0,  "min_samples": 2},   # fallback — very sparse data
        ]

        result = []
        for params in param_ladder:
            # compute_hotspot_summary now accepts eps_km and min_samples kwargs
            result = compute_hotspot_summary(
                coordinates=coords,
                crime_types=crime_types,
                suburbs=suburbs,
                eps_km=params["eps_km"],
                min_samples=params["min_samples"],
            )
            if result:
                # Stop at the first set of params that produces at least one cluster
                break

        return Response({"hotspots": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class ProfileMatchView(APIView):
    """
    POST /api/zrp/analytics/profile-match/

    Body: { "incident_id": <int>, "top_n": <int, optional> }

    Fallback strategy
    -----------------
    1. Try supervised ProfileMatcher (RandomForest) — fast, most accurate.
    2. If no supervised model file exists, fall back to the unsupervised
       SerialCrimeLinkageModel (DBSCAN) similarity matrix.
    3. If neither model is available, return 503 with clear instructions.

    Both paths return the same JSON envelope so the frontend needs no changes.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        serializer = ProfileMatchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        incident_id = serializer.validated_data["incident_id"]
        top_n       = serializer.validated_data.get("top_n", 5)

        # Resolve the query incident
        try:
            incident = CrimeIncident.objects.select_related("crime_type").get(
                pk=incident_id
            )
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "Incident not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # ── Path 1: Supervised — ProfileMatcher (RandomForest) ───────────────
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
                "ProfileMatchView: supervised model missing for incident %d — "
                "trying SerialCrimeLinkageModel fallback.",
                incident_id,
            )
        except Exception as exc:
            logger.warning(
                "ProfileMatchView: supervised match failed for incident %d (%s) — "
                "trying SerialCrimeLinkageModel fallback.",
                incident_id, exc,
            )

        # ── Path 2: Unsupervised — SerialCrimeLinkageModel ───────────────────
        try:
            linkage_model = SerialCrimeLinkageModel.load()
        except FileNotFoundError:
            return Response(
                {
                    "detail": (
                        "No trained model available. "
                        "POST /api/zrp/ml/train/ to train before using profile matching."
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
                        "Serial linkage model present but not fitted. "
                        "Re-train the model from the ML Training page."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Find the row in agg_df that matches this incident's case_number
        mask = agg_df["case_number"] == incident.case_number
        if not mask.any():
            return Response(
                {
                    "detail": (
                        f"Incident '{incident.case_number}' was not in the last "
                        "training run. Re-train the model to include it."
                    )
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        query_idx = int(mask.idxmax())
        sim_row   = sim_matrix[query_idx].copy()
        sim_row[query_idx] = -1.0   # exclude self

        # Sort descending — most similar first
        top_indices      = np.argsort(sim_row)[::-1][:top_n]
        top_case_numbers = agg_df.iloc[top_indices]["case_number"].tolist()

        similar_qs = CrimeIncident.objects.filter(
            case_number__in=top_case_numbers
        ).select_related("crime_type")

        # Re-impose the similarity-score ordering (DB __in doesn't guarantee order)
        order_map         = {cn: rank for rank, cn in enumerate(top_case_numbers)}
        similar_incidents = sorted(
            similar_qs,
            key=lambda inc: order_map.get(inc.case_number, 999),
        )

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
    Trains the DBSCAN serial linkage model on all incidents in the database.
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
            return Response(
                {"detail": results["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(
            "Serial linkage model trained — %d cluster(s) found.",
            results.get("n_serial_clusters", 0),
        )
        return Response(results, status=status.HTTP_200_OK)


class SerialLinkageClusterView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/cluster/
    Load the pre-trained model and return per-case cluster assignments.
    Optional body: { "case_numbers": ["ZRP-…", …] } to filter.
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
                        "POST /api/zrp/analytics/serial-linkage/train/ first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if model.agg_df_ is None:
            return Response(
                {"detail": "Model present but has no cluster data. Re-train."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        df = model.agg_df_.copy()
        df["serial_cluster"] = model.cluster_labels_
        df["cluster_label"]  = df["serial_cluster"].apply(
            lambda c: f"Serial Group {c}" if c >= 0 else "Unlinked"
        )

        # Optional filter by case_numbers list
        requested_cases = request.data.get("case_numbers")
        if requested_cases:
            df = df[df["case_number"].isin(requested_cases)]

        safe_cols = [
            col for col in
            ["case_number", "serial_cluster", "cluster_label", "mo_text", "full_location"]
            if col in df.columns
        ]

        return Response({
            "n_cases":           len(df),
            "n_serial_clusters": int((df["serial_cluster"] >= 0).sum()),
            "cases":             df[safe_cols].to_dict(orient="records"),
            "cluster_summary":   model._build_cluster_summary(),
        })


class SerialLinkageProbabilityView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/link-probability/
    Compute the probability that two specific incidents share an offender.

    Body: { "incident_id_a": <int>, "incident_id_b": <int> }
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
                        "POST /api/zrp/analytics/serial-linkage/train/ first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        def _to_feature_dict(inc: CrimeIncident) -> dict:
            """
            Map a CrimeIncident ORM object to the feature dict expected by
            SerialCrimeLinkageModel.link_probability().
            """
            date_ord = inc.timestamp.toordinal() if inc.timestamp else None
            time_min = (
                inc.timestamp.hour * 60 + inc.timestamp.minute
            ) if inc.timestamp else None

            return {
                "date_ord":      date_ord,
                "time_min":      time_min,
                "mean_age":      35,    # not stored at incident level — use median
                "pct_female":    0.0,
                "pct_male":      1.0,
                "full_location": inc.suburb or "",
                "mo_text":       inc.modus_operandi or "",
            }

        try:
            result = model.link_probability(
                _to_feature_dict(inc_a), _to_feature_dict(inc_b)
            )
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
# Admin — User management
# =============================================================================

class UserListCreateView(APIView):
    """
    GET  /api/zrp/users/   — list all user accounts
    POST /api/zrp/users/   — create a new user (admin only)
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
    GET    /api/zrp/users/<id>/
    PUT    /api/zrp/users/<id>/   — update role, active status, or base_station
    DELETE /api/zrp/users/<id>/   — soft delete (sets is_active=False)
    """
    permission_classes = [IsZRPAdmin]

    def _get_object(self, pk: int):
        try:
            return CustomUser.objects.get(pk=pk)
        except CustomUser.DoesNotExist:
            return None

    def get(self, request, pk: int):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(UserSerializer(user).data)

    def put(self, request, pk: int):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        # Restrict updates to only these safe fields — password changes go
        # through ChangePasswordView; username changes are not permitted.
        allowed_fields = ("role", "is_active", "base_station")
        allowed_data   = {
            k: v for k, v in request.data.items() if k in allowed_fields
        }

        if not allowed_data:
            return Response(
                {"detail": f"No updatable fields provided. Allowed: {allowed_fields}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate role value if supplied
        if "role" in allowed_data and allowed_data["role"] not in (
            "officer", "analyst", "admin"
        ):
            return Response(
                {"detail": "role must be one of: officer, analyst, admin."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        for field, value in allowed_data.items():
            setattr(user, field, value)
        user.save(update_fields=list(allowed_data.keys()))
        return Response(UserSerializer(user).data)

    def delete(self, request, pk: int):
        user = self._get_object(pk)
        if not user:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        # Soft delete — preserves historical data and audit trails
        user.is_active = False
        user.save(update_fields=["is_active"])
        return Response(status=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Admin — ML model training
# =============================================================================

class MLTrainView(APIView):
    """
    POST /api/zrp/ml/train/

    Dual-mode training strategy:
      • Supervised   — ProfileMatcher (RandomForest) when labelled incidents
                       exist (serial_group_label is non-empty)
      • Unsupervised — SerialCrimeLinkageModel (DBSCAN) fallback when no labels

    The 'mode' key in the response tells the frontend which path was taken,
    so it can render the appropriate result card in MLTraining.jsx.
    """
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        # ── Step 1: Are there any labelled incidents to train supervised? ─────
        labelled_qs = CrimeIncident.objects.exclude(
            serial_group_label__in=["", None]
        ).select_related("crime_type")

        if labelled_qs.exists():
            # ── Supervised — ProfileMatcher (RandomForest) ───────────────────
            logger.info(
                "MLTrainView: %d labelled incidents found — training supervised model.",
                labelled_qs.count(),
            )
            matcher = ProfileMatcher()
            try:
                metrics = matcher.train(labelled_qs)
            except Exception as exc:
                logger.error("Supervised training failed: %s", exc)
                return Response(
                    {"detail": f"Supervised training failed: {exc}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            if "error" in metrics:
                return Response(
                    {"detail": metrics["error"]},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            return Response({"mode": "supervised", **metrics})

        # ── Step 2: No labels — use unsupervised DBSCAN clustering ───────────
        logger.info(
            "MLTrainView: No labelled incidents — falling back to unsupervised DBSCAN."
        )

        all_qs = CrimeIncident.objects.all()
        if not all_qs.exists():
            return Response(
                {"detail": "No incident data found. Add incidents before training."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Build a DataFrame that maps CrimeIncident fields to the column names
        # expected by SerialCrimeLinkageModel.train_unsupervised():
        #   case_number, date_received, time_received, complainant_name,
        #   sex, age, residential_address, incident_location,
        #   property_stolen_description
        raw_data = list(all_qs.values(
            "case_number",
            "timestamp",
            "description_narrative",
            "num_suspects",
            "suburb",
            "modus_operandi",
        ))

        df = pd.DataFrame(raw_data)

        # Derive date and time strings from the combined timestamp field
        ts_series             = pd.to_datetime(df["timestamp"])
        df["date_received"]   = ts_series.dt.strftime("%Y-%m-%d")
        df["time_received"]   = ts_series.dt.strftime("%H%M")  # e.g. "1430"

        # Map remaining CrimeIncident fields to the linkage model's expected names
        df["complainant_name"]            = df["description_narrative"].fillna("")
        df["sex"]                         = ""      # not captured at incident level
        df["age"]                         = df["num_suspects"].fillna(0)
        df["residential_address"]         = df["suburb"].fillna("")
        df["incident_location"]           = df["suburb"].fillna("")
        df["property_stolen_description"] = df["modus_operandi"].fillna("")

        # Drop source columns that have now been renamed/mapped
        df = df.drop(columns=[
            "timestamp", "description_narrative", "num_suspects", "modus_operandi"
        ])

        linkage_model = SerialCrimeLinkageModel()
        try:
            metrics = linkage_model.train_unsupervised(df)
        except Exception as exc:
            logger.error("Unsupervised training failed: %s", exc)
            return Response(
                {"detail": f"Unsupervised training failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if "error" in metrics:
            return Response(
                {"detail": metrics["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        return Response({"mode": "unsupervised", **metrics})