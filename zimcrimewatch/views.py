"""
zimcrimewatch/views.py
======================
ZimCrimeWatch — API Views (APIView only, no ViewSets)

Serves both the Flutter mobile app (public, anonymised) and the
React ZRP dashboard (authenticated, full data + analytics).

Endpoint groups
---------------
Public (no auth)
  POST /api/public/auth/login/
  POST /api/public/auth/logout/
  GET  /api/public/crimes/           — anonymised map pins for Flutter
  GET  /api/public/crime-types/      — list of crime categories + icons

ZRP Dashboard (authentication required)
  GET/POST         /api/zrp/incidents/
  GET/PUT/DELETE   /api/zrp/incidents/<id>/
  GET              /api/zrp/incidents/<id>/similar/    ← ProfileMatcher
  GET              /api/zrp/dashboard/summary/
  GET/POST         /api/zrp/crime-types/
  GET/PUT/DELETE   /api/zrp/crime-types/<id>/

Analytics (authentication required)
  GET/POST  /api/zrp/analytics/heatmap/
  GET/POST  /api/zrp/analytics/timeseries/
  GET/POST  /api/zrp/analytics/hotspots/
  POST      /api/zrp/analytics/profile-match/

Serial Crime Linkage  ← NEW (authentication required)
  POST      /api/zrp/analytics/serial-linkage/train/
  POST      /api/zrp/analytics/serial-linkage/cluster/
  POST      /api/zrp/analytics/serial-linkage/link-probability/

Admin only
  GET/POST         /api/zrp/users/
  GET/PUT/DELETE   /api/zrp/users/<id>/
  POST             /api/zrp/ml/train/
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

# ── Internal ML helpers ───────────────────────────────────────────────────────
from .ml_utils import (
    ProfileMatcher,
    compute_hotspot_summary,
    compute_kde_heatmap,
    compute_time_series,
)

# ── Serial Crime Linkage model (new integration) ──────────────────────────────
from .serial_crime_linkage import SerialCrimeLinkageModel

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


# =============================================================================
# Authentication
# =============================================================================

class LoginView(APIView):
    """
    POST /api/public/auth/login/
    Accepts {username, password} and returns JWT access + refresh tokens.
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

        # Generate a fresh JWT pair for this user.
        refresh = RefreshToken.for_user(user)
        logger.info("User %s logged in.", user.username)
        return Response(
            {
                "access":  str(refresh.access_token),
                "refresh": str(refresh),
                "user":    UserSerializer(user).data,
            }
        )


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
            refresh     = RefreshToken(serializer.validated_data["refresh"])
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
    DELETE /api/zrp/incidents/<id>/
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
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(CrimeIncidentSerializer(incident).data)

    def put(self, request, pk):
        incident = self._get_object(pk)
        if not incident:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        serializer = CrimeIncidentSerializer(
            incident, data=request.data, partial=True,
            context={"request": request},
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        # Hard delete restricted to admins.
        if not request.user.zrp_profile.role == "admin":
            return Response(
                {"detail": "Admin role required to delete incidents."},
                status=status.HTTP_403_FORBIDDEN,
            )
        incident = self._get_object(pk)
        if not incident:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        incident.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class IncidentSimilarCasesView(APIView):
    """
    GET /api/zrp/incidents/<id>/similar/
    Returns the top-N most similar cases using the ProfileMatcher ML model.
    Optional query param: ?top_n=5 (default 5)

    Uses ProfileMatcher.find_similar() which was previously missing and
    has now been implemented in ml_utils.py.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def get(self, request, pk):
        # Fetch the incident whose similar cases we want.
        try:
            incident = CrimeIncident.objects.select_related(
                "crime_type"
            ).get(pk=pk)
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )

        top_n = int(request.query_params.get("top_n", 5))

        # ProfileMatcher.find_similar() loads the persisted model and returns
        # a list of CrimeIncident PKs ranked by similarity.
        matcher = ProfileMatcher()
        try:
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

        return Response(
            {
                "source_incident": pk,
                "similar_cases":   CrimeIncidentSerializer(similar_qs, many=True).data,
            }
        )


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
    """
    GET  /api/zrp/crime-types/  — list all crime types with incident counts
    POST /api/zrp/crime-types/  — create a crime type (analyst/admin only)
    """
    permission_classes = [IsZRPAuthenticated]

    def get(self, request):
        qs = CrimeType.objects.annotate(
            incident_count=Count("incidents")
        ).order_by("name")
        return Response(CrimeTypeSerializer(qs, many=True).data)

    def post(self, request):
        if request.user.zrp_profile.role not in ("analyst", "admin"):
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
    """
    GET/PUT/DELETE /api/zrp/crime-types/<id>/
    """
    permission_classes = [IsZRPAuthenticated]

    def _get_ct(self, pk):
        try:
            return CrimeType.objects.annotate(
                incident_count=Count("incidents")
            ).get(pk=pk)
        except CrimeType.DoesNotExist:
            return None

    def get(self, request, pk):
        obj = self._get_ct(pk)
        if not obj:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(CrimeTypeSerializer(obj).data)

    def put(self, request, pk):
        obj = self._get_ct(pk)
        if not obj:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        serializer = CrimeTypeSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        if request.user.zrp_profile.role != "admin":
            return Response(
                {"detail": "Admin role required."},
                status=status.HTTP_403_FORBIDDEN,
            )
        obj = self._get_ct(pk)
        if not obj:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        # Prevent deletion if incidents are already linked to this type.
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
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        d = serializer.validated_data

        qs = CrimeIncident.objects.exclude(location__isnull=True)
        if d.get("crime_type_id"):
            qs = qs.filter(crime_type_id=d["crime_type_id"])
        if d.get("start_date"):
            qs = qs.filter(timestamp__date__gte=d["start_date"])
        if d.get("end_date"):
            qs = qs.filter(timestamp__date__lte=d["end_date"])

        # Extract (lat, lng) tuples from the PostGIS PointField.
        coords = [
            (inc.location.y, inc.location.x)
            for inc in qs.only("location")
            if inc.location
        ]
        if not coords:
            return Response({"heatmap_data": []})

        result = compute_kde_heatmap(
            coords, bandwidth=d.get("bandwidth", 0.01)
        )
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
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
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

        result = compute_time_series(df, d.get("freq", "W"))
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

        # Unpack the PostGIS PointField plus metadata needed for the summary.
        coords      = []
        crime_types = []
        suburbs     = []
        for inc in qs.select_related("crime_type").only(
            "location", "crime_type__name", "suburb"
        ):
            if inc.location:
                coords.append((inc.location.y, inc.location.x))
                crime_types.append(
                    inc.crime_type.name if inc.crime_type else "Unknown"
                )
                suburbs.append(inc.suburb or "")

        if not coords:
            return Response({"hotspots": []})

        result = compute_hotspot_summary(coords, crime_types, suburbs)
        return Response({"hotspots": result})

    def get(self, request):
        return self._run(request)

    def post(self, request):
        return self._run(request)


class ProfileMatchView(APIView):
    """
    POST /api/zrp/analytics/profile-match/
    Body: { "incident_id": <int>, "top_n": <int, optional> }
    Returns the top_n most similar incidents for the given incident_id.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        serializer = ProfileMatchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            incident = CrimeIncident.objects.get(
                pk=serializer.validated_data["incident_id"]
            )
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "Incident not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        top_n   = serializer.validated_data.get("top_n", 5)
        matcher = ProfileMatcher()
        try:
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
            logger.error("ProfileMatcher error: %s", exc)
            return Response(
                {"detail": "Profile matching unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        similar_qs = CrimeIncident.objects.filter(
            pk__in=similar_ids
        ).select_related("crime_type")
        return Response(
            {
                "query_incident": serializer.validated_data["incident_id"],
                "matches":        CrimeIncidentSerializer(similar_qs, many=True).data,
            }
        )


# =============================================================================
# Serial Crime Linkage  ← NEW  (integrates serial_crime_linkage.py)
# =============================================================================

class SerialLinkageTrainView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/train/

    Trains the SerialCrimeLinkageModel in unsupervised mode (DBSCAN) using
    ALL incidents currently in the database.

    No request body required. Returns cluster summary with serial groups.

    Requires analyst or admin role — training is an expensive operation.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        # Pull all incidents — the serial linkage model aggregates them
        # internally by case_number before clustering.
        qs = CrimeIncident.objects.all()
        if not qs.exists():
            return Response(
                {"detail": "No incidents in the database to train on."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Instantiate a fresh model and run unsupervised training.
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
                {"detail": results["error"]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            "Serial linkage model trained: %d clusters found.",
            results.get("n_serial_clusters", 0),
        )
        return Response(results, status=status.HTTP_200_OK)


class SerialLinkageClusterView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/cluster/

    Load the pre-trained serial linkage model and return per-incident
    cluster assignments (serial_cluster, cluster_label, max_similarity_score,
    most_similar_case) for all incidents.

    The model must be trained first via SerialLinkageTrainView.
    Optionally accepts { "case_numbers": [1,2,3] } to filter results.
    """
    permission_classes = [IsZRPAnalystOrAdmin]

    def post(self, request):
        # Load the persisted model — raises FileNotFoundError if not trained.
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

        # Optional filter: only return clusters for specific case numbers.
        requested_cases = request.data.get("case_numbers")

        if model.agg_df_ is None:
            return Response(
                {"detail": "Model is trained but has no cluster data. Re-train."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Convert the aggregated DataFrame to a dict for JSON serialisation.
        df = model.agg_df_.copy()
        df["serial_cluster"] = model.cluster_labels_

        # Map numeric cluster IDs to human-readable labels.
        df["cluster_label"] = df["serial_cluster"].apply(
            lambda c: f"Serial Group {c}" if c >= 0 else "Unlinked"
        )

        if requested_cases:
            # Filter to only the requested case numbers.
            df = df[df["case_number"].isin(requested_cases)]

        # Build the response payload — only include JSON-safe columns.
        safe_cols = [
            "case_number", "serial_cluster", "cluster_label",
            "mo_text", "full_location",
        ]
        # Keep only columns that actually exist in the DataFrame.
        existing_cols = [c for c in safe_cols if c in df.columns]

        return Response(
            {
                "n_cases":          len(df),
                "n_serial_clusters": int(
                    (df["serial_cluster"] >= 0).sum()
                ),
                "cases":            df[existing_cols].to_dict(orient="records"),
                "cluster_summary":  model._build_cluster_summary(),
            }
        )


class SerialLinkageProbabilityView(APIView):
    """
    POST /api/zrp/analytics/serial-linkage/link-probability/

    Compute the probability that two specific incidents were committed by
    the same offender.

    Request body
    ------------
    {
      "incident_id_a": <int>,
      "incident_id_b": <int>
    }

    Response
    --------
    {
      "composite_similarity": 0.72,
      "link_probability":     0.84,
      "verdict":              "HIGH – Likely same offender",
      "feature_scores": {
          "temporal_similarity": 0.90,
          "spatial_similarity":  0.85,
          "mo_similarity":       0.60,
          "age_similarity":      0.70,
          "gender_similarity":   1.00
      }
    }
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

        # Fetch both incidents from the DB.
        try:
            inc_a = CrimeIncident.objects.select_related("crime_type").get(pk=id_a)
            inc_b = CrimeIncident.objects.select_related("crime_type").get(pk=id_b)
        except CrimeIncident.DoesNotExist:
            return Response(
                {"detail": "One or both incidents not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Load the trained serial linkage model.
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

        # Build the feature dicts expected by SerialCrimeLinkageModel.link_probability().
        # Each dict must contain the aggregated case-level features.
        def _incident_to_feature_dict(inc) -> dict:
            """
            Convert a CrimeIncident ORM object into the case-feature dict
            expected by SerialCrimeLinkageModel.link_probability().

            The serial linkage model was designed around the Zimbabwe Police
            occurrence book schema, so we map Django model fields accordingly.
            """
            import datetime as dt

            # Convert date to an ordinal number (days since year 1) for
            # arithmetic comparisons inside the linkage model.
            date_ord = (
                inc.timestamp.toordinal()
                if inc.timestamp
                else None
            )

            # Convert time to minutes-since-midnight for temporal comparison.
            if inc.timestamp:
                time_min = (
                    inc.timestamp.hour * 60 + inc.timestamp.minute
                )
            else:
                time_min = None

            return {
                "date_ord":   date_ord,
                "time_min":   time_min,
                "mean_age":   35,              # single incident → no age aggregation
                "pct_female": 0.0,
                "pct_male":   1.0,
                "full_location": (
                    f"{inc.suburb or ''} {inc.incident_location or ''}"
                ).strip(),
                "mo_text":    inc.modus_operandi or "",
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

        # Enrich the response with the incident IDs for easy reference.
        result["incident_id_a"] = id_a
        result["incident_id_b"] = id_b
        return Response(result)


# =============================================================================
# Admin — ML training trigger (ProfileMatcher)
# =============================================================================

class MLTrainView(APIView):
    """
    POST /api/zrp/ml/train/
    Re-train the RandomForest ProfileMatcher model.
    Requires admin role.
    """
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        # Use labelled incidents (those with a serial_group_label) for training.
        qs = CrimeIncident.objects.exclude(serial_group_label="").select_related(
            "crime_type"
        )
        if not qs.exists():
            return Response(
                {
                    "detail": (
                        "No labelled incidents found. "
                        "Set 'serial_group_label' on incidents before training."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        matcher = ProfileMatcher()
        try:
            metrics = matcher.train(qs)
        except Exception as exc:
            logger.error("MLTrainView training failed: %s", exc)
            return Response(
                {"detail": f"Training failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if "error" in metrics:
            return Response(
                {"detail": metrics["error"]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(metrics, status=status.HTTP_200_OK)


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
            return Response(
                UserSerializer(user).data, status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserDetailView(APIView):
    """
    GET/PUT/DELETE /api/zrp/users/<id>/
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
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(UserSerializer(user).data)

    def put(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        # Only allow updating safe fields — never username or password here.
        allowed_fields = ("role", "is_active", "base_station")
        allowed_data = {
            k: v for k, v in request.data.items() if k in allowed_fields
        }
        for field, value in allowed_data.items():
            setattr(user, field, value)
        user.save(update_fields=list(allowed_data.keys()))
        return Response(UserSerializer(user).data)

    def delete(self, request, pk):
        user = self._get_object(pk)
        if not user:
            return Response(
                {"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND
            )
        # Soft delete — preserves historical audit data.
        user.is_active = False
        user.save(update_fields=["is_active"])
        return Response(status=status.HTTP_204_NO_CONTENT)
