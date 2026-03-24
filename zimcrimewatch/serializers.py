"""
ZimCrimeWatch — DRF Serializers
================================
Handles serialization / deserialization for all models and API payloads.

PostGIS note
------------
CrimeIncident stores coordinates in a single PostGIS PointField (SRID 4326).
  • Read  → latitude  = location.y,  longitude = location.x
  • Write → clients send plain {latitude, longitude} floats; validate() assembles
            them into a django.contrib.gis.geos.Point and assigns location.
"""
from __future__ import annotations

from django.contrib.auth.password_validation import validate_password
from django.contrib.gis.geos import Point
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import CrimeIncident, CrimeType, CustomUser as User


# =============================================================================
# Auth serializers
# =============================================================================

class LoginSerializer(serializers.Serializer):
    """
    Accepts a ZRP badge number + password.
    The view uses authenticate(username=zrp_badge_number, …) because
    CustomUser.USERNAME_FIELD = 'zrp_badge_number'.
    """
    zrp_badge_number = serializers.CharField(
        label="ZRP Badge Number",
        help_text="Officer's unique ZRP badge number (e.g. '1234')",
    )
    password = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
    )


class TokenRefreshSerializer(serializers.Serializer):
    """Wraps the simplejwt refresh token string for the logout / refresh endpoints."""
    refresh = serializers.CharField()


# =============================================================================
# Registration serializer
# =============================================================================

class RegisterUserSerializer(serializers.Serializer):
    """
    Public registration endpoint — used for self-registration of ZRP officers.

    Role assignment rules:
      • The default role for any self-registered user is 'officer'.
      • Only an existing 'admin' user can promote someone to 'analyst' or 'admin'
        via the /zrp/users/<id>/ PUT endpoint (handled in UserDetailView).
      • This prevents privilege escalation via the registration endpoint.
    """
    username         = serializers.CharField(max_length=20)
    first_name       = serializers.CharField(max_length=20)
    last_name        = serializers.CharField(max_length=20)
    zrp_badge_number = serializers.CharField(max_length=20)
    password         = serializers.CharField(
        write_only=True,
        min_length=8,
        style={"input_type": "password"},
    )
    password_confirm = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
        label="Confirm Password",
    )

    def validate_username(self, value: str) -> str:
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                "A user with this username already exists."
            )
        return value

    def validate_zrp_badge_number(self, value: str) -> str:
        if User.objects.filter(zrp_badge_number=value).exists():
            raise serializers.ValidationError(
                "This badge number is already registered."
            )
        return value

    def validate_password(self, value: str) -> str:
        """Run Django's built-in password validators."""
        validate_password(value)
        return value

    def validate(self, attrs: dict) -> dict:
        """Confirm the two password fields match before creating the user."""
        if attrs.get("password") != attrs.get("password_confirm"):
            raise serializers.ValidationError(
                {"password_confirm": "Password fields did not match."}
            )
        return attrs

    def create(self, validated_data: dict) -> User:
        # Remove the confirmation field before passing to create_user()
        validated_data.pop("password_confirm")
        password = validated_data.pop("password")

        # All self-registered users start as 'officer' — safest default
        user = User.objects.create_user(
            username=validated_data["username"],
            first_name=validated_data["first_name"],
            last_name=validated_data["last_name"],
            zrp_badge_number=validated_data["zrp_badge_number"],
            password=password,
            role="officer",  # self-registered users cannot choose their role
        )
        return user


# =============================================================================
# Password reset serializers
# =============================================================================

class ForgotPasswordSerializer(serializers.Serializer):
    """
    Step 1 — Accept a badge number and initiate a password reset.
    In production this would send an email with a tokenised reset link.
    For the prototype the token is returned directly in the response so
    the frontend can immediately call the reset endpoint.
    """
    zrp_badge_number = serializers.CharField(
        label="ZRP Badge Number",
        help_text="The badge number you registered with.",
    )


class ResetPasswordSerializer(serializers.Serializer):
    """
    Step 2 — Accept the reset token + new password and change the password.
    """
    zrp_badge_number = serializers.CharField()
    token            = serializers.CharField(
        help_text="The one-time reset token received from the forgot-password step."
    )
    new_password     = serializers.CharField(
        write_only=True,
        min_length=8,
        style={"input_type": "password"},
    )
    confirm_password = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
    )

    def validate_new_password(self, value: str) -> str:
        validate_password(value)
        return value

    def validate(self, attrs: dict) -> dict:
        if attrs.get("new_password") != attrs.get("confirm_password"):
            raise serializers.ValidationError(
                {"confirm_password": "Password fields did not match."}
            )
        return attrs


class ChangePasswordSerializer(serializers.Serializer):
    """
    Allows an authenticated user to change their own password by providing
    the current password for verification.
    """
    current_password = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
    )
    new_password     = serializers.CharField(
        write_only=True,
        min_length=8,
        style={"input_type": "password"},
    )
    confirm_password = serializers.CharField(
        write_only=True,
        style={"input_type": "password"},
    )

    def validate_new_password(self, value: str) -> str:
        validate_password(value)
        return value

    def validate(self, attrs: dict) -> dict:
        if attrs.get("new_password") != attrs.get("confirm_password"):
            raise serializers.ValidationError(
                {"confirm_password": "Password fields did not match."}
            )
        return attrs


# =============================================================================
# User serializers
# =============================================================================

class UserSerializer(serializers.ModelSerializer):
    """Read-only serializer — safe public fields, no password exposed."""

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
        read_only_fields = fields


class CreateUserSerializer(serializers.Serializer):
    """
    Admin-only write serializer — creates a new CustomUser with a specified role.
    Unlike RegisterUserSerializer, this allows admin users to set any role.
    """

    username         = serializers.CharField(max_length=20)
    first_name       = serializers.CharField(max_length=20)
    last_name        = serializers.CharField(max_length=20)
    zrp_badge_number = serializers.CharField(max_length=20)
    password         = serializers.CharField(write_only=True, min_length=8)
    role             = serializers.ChoiceField(
        choices=["analyst", "officer", "admin"]
    )
    base_station     = serializers.IntegerField(
        required=False, allow_null=True, default=None
    )

    def validate_username(self, value: str) -> str:
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                "A user with this username already exists."
            )
        return value

    def validate_zrp_badge_number(self, value: str) -> str:
        if User.objects.filter(zrp_badge_number=value).exists():
            raise serializers.ValidationError(
                "This badge number is already registered."
            )
        return value

    def create(self, validated_data: dict) -> User:
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
    # Populated by the annotate(incident_count=Count("incidents")) in the view
    incident_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model  = CrimeType
        fields = ["id", "name", "description", "icon", "incident_count"]


# =============================================================================
# Crime Incident serializers
# =============================================================================

class CrimeIncidentSerializer(serializers.ModelSerializer):
    """
    Full serializer for ZRP dashboard users (authenticated).

    Read  → latitude / longitude derived from the PostGIS PointField
    Write → clients send plain float {latitude, longitude};
            validate() builds the PostGIS Point and assigns it to location.
    """

    # ── Virtual read-only fields derived from the PostGIS PointField ──────────
    latitude  = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()

    # ── Write-only input fields — accepted by the deserialiser ────────────────
    # source="latitude" / source="longitude" causes DRF to pop these values into
    # attrs["latitude"] / attrs["longitude"] during validation, which we then
    # intercept in validate() to build the PostGIS Point.
    latitude_input  = serializers.FloatField(
        write_only=True, source="latitude",  required=False
    )
    longitude_input = serializers.FloatField(
        write_only=True, source="longitude", required=False
    )

    crime_type_name     = serializers.CharField(source="crime_type.name", read_only=True)
    created_by_username = serializers.CharField(source="created_by.username", read_only=True)

    class Meta:
        model  = CrimeIncident
        fields = [
            "id",
            "case_number",
            "crime_type",
            "crime_type_name",
            "timestamp",
            "latitude",        # read-only — derived from PostGIS
            "longitude",       # read-only — derived from PostGIS
            "latitude_input",  # write-only — accepted from client
            "longitude_input", # write-only — accepted from client
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
        read_only_fields = [
            "time_of_day", "day_of_week", "updated_at", "created_at"
        ]

    def get_latitude(self, obj) -> float | None:
        """Return WGS-84 latitude (Y coordinate) from the PostGIS PointField."""
        return obj.location.y if obj.location else None

    def get_longitude(self, obj) -> float | None:
        """Return WGS-84 longitude (X coordinate) from the PostGIS PointField."""
        return obj.location.x if obj.location else None

    def validate_case_number(self, value: str) -> str:
        """Ensure case_number is unique across all incidents (except self on update)."""
        qs = CrimeIncident.objects.filter(case_number=value)
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("Case number already exists.")
        return value

    def validate(self, attrs: dict) -> dict:
        """
        Assemble the PostGIS Point from the latitude_input / longitude_input fields.
        The source= mapping on those fields pops them into attrs["latitude"] /
        attrs["longitude"] before this method is called.

        On PATCH requests both coordinate fields are optional; we only update
        location when both are provided together.
        """
        lat = attrs.pop("latitude", None)   # placed here by source="latitude"
        lon = attrs.pop("longitude", None)  # placed here by source="longitude"

        if lat is not None and lon is not None:
            # PostGIS Point takes (x=longitude, y=latitude) — note the axis order
            attrs["location"] = Point(lon, lat, srid=4326)
        elif lat is not None or lon is not None:
            # Partial coordinate is invalid — both must be supplied together
            raise serializers.ValidationError(
                "Both `latitude` and `longitude` must be provided together."
            )
        # If neither is provided on a PATCH request, leave location unchanged
        return attrs


class PublicCrimeIncidentSerializer(serializers.ModelSerializer):
    """
    Anonymised serializer for the public Flutter mobile app.

    Strips all sensitive fields (descriptions, case numbers, officer notes).
    Rounds coordinates to 4 decimal places (~11 m precision) so the exact
    victim address cannot be determined by end-users.
    """

    latitude        = serializers.SerializerMethodField()
    longitude       = serializers.SerializerMethodField()
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
# Analytics request serializers
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
        choices=["D", "W", "M"],
        default="W",
        help_text="D = daily, W = weekly, M = monthly",
    )


class ProfileMatchRequestSerializer(serializers.Serializer):
    incident_id = serializers.IntegerField()
    top_n       = serializers.IntegerField(default=5, min_value=1, max_value=50)