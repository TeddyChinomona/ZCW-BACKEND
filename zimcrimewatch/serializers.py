"""
ZimCrimeWatch - DRF Serializers
Handles serialization / deserialization for all models and API payloads.

PostGIS note
------------
CrimeIncident no longer has separate `latitude` / `longitude` DB columns.
All coordinate data lives in the PostGIS `location` PointField.
  • Read  → `latitude`  = SerializerMethodField pulling location.y
            `longitude` = SerializerMethodField pulling location.x
  • Write → clients send plain `latitude` + `longitude` floats; `validate()`
            assembles them into a django.contrib.gis.geos.Point and sets
            `location` on the validated data.
"""
from __future__ import annotations

from django.contrib.gis.geos import Point
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import CrimeIncident, CrimeType, CustomUser as User


# =============================================================================
# Auth serializers
# =============================================================================

class LoginSerializer(serializers.Serializer):
    """
    Accepts ZRP badge number + password.
    The view passes `zrp_badge_number` to `authenticate()` because
    CustomUser.USERNAME_FIELD = 'zrp_badge_number'.
    """
    zrp_badge_number = serializers.CharField(
        label="ZRP Badge Number",
        help_text="Officer's unique ZRP badge number (e.g. 'ZRP-001234')",
    )
    password = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
    )


class TokenRefreshSerializer(serializers.Serializer):
    """Used by TokenRefreshView — accepts a simplejwt refresh token string."""
    refresh = serializers.CharField()


# =============================================================================
# User serializers
# =============================================================================

class UserSerializer(serializers.ModelSerializer):
    """Read serializer — safe fields only, no password."""

    class Meta:
        model  = User
        fields = [
            "id",
            "username",
            "fullname",
            "first_name",
            "last_name",
            "zrp_badge_number",
            "role",
            "is_active",
            "base_station",
        ]
        read_only_fields = fields  # this serializer is read-only


class CreateUserSerializer(serializers.Serializer):
    """Write serializer — validates and creates a new CustomUser."""

    username         = serializers.CharField(max_length=20)
    first_name       = serializers.CharField(max_length=20)
    last_name        = serializers.CharField(max_length=20)
    zrp_badge_number = serializers.CharField(max_length=20)
    password         = serializers.CharField(write_only=True, min_length=8)
    role             = serializers.ChoiceField(choices=["analyst", "officer", "admin"])
    base_station     = serializers.IntegerField(required=False, allow_null=True, default=None)

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value

    def validate_zrp_badge_number(self, value):
        if User.objects.filter(zrp_badge_number=value).exists():
            raise serializers.ValidationError("This badge number is already registered.")
        return value

    def create(self, validated_data):
        base_station_id = validated_data.pop("base_station", None)
        password        = validated_data.pop("password")
        user = User.objects.create_user(password=password, **validated_data)
        if base_station_id:
            user.base_station_id = base_station_id
            user.save(update_fields=["base_station"])
        return user


# =============================================================================
# Crime Type serializer
# =============================================================================

class CrimeTypeSerializer(serializers.ModelSerializer):
    incident_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model  = CrimeType
        fields = ["id", "name", "description", "icon", "incident_count"]


# =============================================================================
# Crime Incident serializers
# =============================================================================

class CrimeIncidentSerializer(serializers.ModelSerializer):
    """
    Full serializer for ZRP dashboard users.

    Read fields
    -----------
    latitude  — derived from location.y  (PostGIS Point Y)
    longitude — derived from location.x  (PostGIS Point X)

    Write fields
    ------------
    latitude  — float, required when creating / updating
    longitude — float, required when creating / updating
    The `validate()` method builds the PostGIS Point and sets `location`.
    The raw `location` field is excluded from the serializer output so
    clients always work with plain floats.
    """

    # ── Virtual read fields ───────────────────────────────────────────────────
    latitude  = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()

    # ── Virtual write fields (accepted on input, not a real model field here) ─
    latitude_input  = serializers.FloatField(write_only=True, source="latitude",  required=False)
    longitude_input = serializers.FloatField(write_only=True, source="longitude", required=False)

    crime_type_name      = serializers.CharField(source="crime_type.name", read_only=True)
    created_by_username  = serializers.CharField(source="created_by.username", read_only=True)

    class Meta:
        model  = CrimeIncident
        fields = [
            "id",
            "case_number",
            "crime_type",
            "crime_type_name",
            "timestamp",
            # spatial — read as floats, write as floats
            "latitude",
            "longitude",
            "latitude_input",
            "longitude_input",
            "suburb",
            "description_narrative",
            "modus_operandi",
            "status",
            "weapon_used",
            "num_suspects",
            "time_of_day",
            "day_of_week",
            "serial_group_label",
            "created_by",
            "created_by_username",
            "updated_at",
            "created_at",
        ]
        read_only_fields = ["time_of_day", "day_of_week", "updated_at", "created_at"]

    # ── Getters for the read fields ──────────────────────────────────────────

    def get_latitude(self, obj) -> float | None:
        return obj.location.y if obj.location else None

    def get_longitude(self, obj) -> float | None:
        return obj.location.x if obj.location else None

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_case_number(self, value):
        qs = CrimeIncident.objects.filter(case_number=value)
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("Case number already exists.")
        return value

    def validate(self, attrs):
        """
        Build the PostGIS Point from latitude_input / longitude_input.
        On partial updates (PATCH) the fields are optional; we only
        update `location` when at least one coordinate is supplied.
        """
        lat = attrs.pop("latitude",  None)   # from latitude_input source mapping
        lon = attrs.pop("longitude", None)   # from longitude_input source mapping

        if lat is not None and lon is not None:
            attrs["location"] = Point(lon, lat, srid=4326)  # Point(x=lon, y=lat)
        elif lat is not None or lon is not None:
            raise serializers.ValidationError(
                "Both `latitude` and `longitude` must be provided together."
            )
        # else: no coordinate change — leave `location` untouched (PATCH)
        return attrs


class PublicCrimeIncidentSerializer(serializers.ModelSerializer):
    """
    Anonymized serializer for the public Flutter mobile app.

    Strips all sensitive fields (descriptions, M.O., case numbers).
    Rounds coordinates to 4 dp (~11 m precision) to prevent exact
    victim-location identification.
    """

    latitude  = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()
    crime_type_name = serializers.CharField(source="crime_type.name", read_only=True)

    class Meta:
        model  = CrimeIncident
        fields = [
            "id",
            "crime_type",
            "crime_type_name",
            "timestamp",
            "latitude",
            "longitude",
            "suburb",
            "time_of_day",
            "day_of_week",
        ]

    def get_latitude(self, obj) -> float | None:
        return round(obj.location.y, 4) if obj.location else None

    def get_longitude(self, obj) -> float | None:
        return round(obj.location.x, 4) if obj.location else None


# =============================================================================
# Analytics request serializers (unchanged)
# =============================================================================

class HeatmapRequestSerializer(serializers.Serializer):
    crime_type_id = serializers.IntegerField(required=False, allow_null=True)
    start_date    = serializers.DateField(required=False, allow_null=True)
    end_date      = serializers.DateField(required=False, allow_null=True)
    bandwidth     = serializers.FloatField(required=False, default=0.01)


class TimeSeriesRequestSerializer(serializers.Serializer):
    crime_type_id = serializers.IntegerField(required=False, allow_null=True)
    start_date    = serializers.DateField(required=False, allow_null=True)
    end_date      = serializers.DateField(required=False, allow_null=True)
    freq          = serializers.ChoiceField(
        choices=["D", "W", "M"], default="W",
        help_text="D = daily, W = weekly, M = monthly"
    )


class ProfileMatchRequestSerializer(serializers.Serializer):
    incident_id   = serializers.IntegerField()
    top_n         = serializers.IntegerField(default=5, min_value=1, max_value=50)
