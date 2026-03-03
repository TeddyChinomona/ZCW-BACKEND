"""
ZimCrimeWatch - DRF Serializers
Handles serialization/deserialization for all models and API payloads.
"""
from django.contrib.auth.models import User
from rest_framework import serializers
from .models import CrimeType, ZRPProfile, CrimeIncident


# ---------------------------------------------------------------------------
# Auth serializers
# ---------------------------------------------------------------------------

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True, style={"input_type": "password"})


class UserSerializer(serializers.ModelSerializer):
    role = serializers.SerializerMethodField()
    full_name = serializers.SerializerMethodField()
    badge_number = serializers.SerializerMethodField()
    station = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "email", "is_active", "role", "full_name", "badge_number", "station"]

    def get_role(self, obj):
        return obj.zrp_profile.role if hasattr(obj, "zrp_profile") else None

    def get_full_name(self, obj):
        return obj.zrp_profile.full_name if hasattr(obj, "zrp_profile") else obj.get_full_name()

    def get_badge_number(self, obj):
        return obj.zrp_profile.badge_number if hasattr(obj, "zrp_profile") else ""

    def get_station(self, obj):
        return obj.zrp_profile.station if hasattr(obj, "zrp_profile") else ""


class CreateUserSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)
    full_name = serializers.CharField(max_length=255)
    badge_number = serializers.CharField(max_length=50, required=False, default="")
    station = serializers.CharField(max_length=150, required=False, default="")
    role = serializers.ChoiceField(choices=["analyst", "officer", "admin"])

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value


# ---------------------------------------------------------------------------
# Crime type serializer
# ---------------------------------------------------------------------------

class CrimeTypeSerializer(serializers.ModelSerializer):
    incident_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = CrimeType
        fields = ["id", "name", "description", "icon", "incident_count"]


# ---------------------------------------------------------------------------
# Crime Incident serializers
# ---------------------------------------------------------------------------

class CrimeIncidentSerializer(serializers.ModelSerializer):
    crime_type_name = serializers.CharField(source="crime_type.name", read_only=True)
    created_by_username = serializers.CharField(source="created_by.username", read_only=True)

    class Meta:
        model = CrimeIncident
        fields = [
            "id", "case_number", "crime_type", "crime_type_name",
            "timestamp", "latitude", "longitude", "suburb",
            "description_narrative", "modus_operandi",
            "status", "weapon_used", "num_suspects",
            "time_of_day", "day_of_week", "serial_group_label",
            "created_by", "created_by_username", "updated_at", "created_at",
        ]
        read_only_fields = ["time_of_day", "day_of_week", "updated_at", "created_at"]

    def validate_case_number(self, value):
        request = self.context.get("request")
        qs = CrimeIncident.objects.filter(case_number=value)
        # Exclude current instance when updating
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("Case number already exists.")
        return value


class PublicCrimeIncidentSerializer(serializers.ModelSerializer):
    """
    Anonymized serializer for the public Flutter mobile app.
    Strips sensitive fields (descriptions, MO, case numbers).
    Rounds coordinates to 4dp (~11m precision) to prevent exact victim location identification.
    """
    crime_type_name = serializers.CharField(source="crime_type.name", read_only=True)
    crime_type_icon = serializers.CharField(source="crime_type.icon", read_only=True)
    latitude = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()

    class Meta:
        model = CrimeIncident
        fields = [
            "id", "crime_type_name", "crime_type_icon",
            "timestamp", "latitude", "longitude",
            "suburb", "time_of_day", "day_of_week",
        ]

    def get_latitude(self, obj):
        return round(float(obj.latitude), 4)

    def get_longitude(self, obj):
        return round(float(obj.longitude), 4)


# ---------------------------------------------------------------------------
# Analytics payload serializers (for request validation)
# ---------------------------------------------------------------------------

class HeatmapRequestSerializer(serializers.Serializer):
    crime_type_id = serializers.IntegerField(required=False, allow_null=True)
    start_date = serializers.DateField(required=False, allow_null=True)
    end_date = serializers.DateField(required=False, allow_null=True)
    bandwidth = serializers.FloatField(required=False, default=0.01, min_value=0.001, max_value=1.0)
    grid_size = serializers.IntegerField(required=False, default=50, min_value=10, max_value=200)
    bounds = serializers.DictField(required=False, child=serializers.FloatField(), allow_null=True,
                                   help_text="{'min_lat': ..., 'max_lat': ..., 'min_lng': ..., 'max_lng': ...}")


class TimeSeriesRequestSerializer(serializers.Serializer):
    PERIOD_CHOICES = ["daily", "weekly", "monthly"]
    crime_type_id = serializers.IntegerField(required=False, allow_null=True)
    start_date = serializers.DateField(required=False, allow_null=True)
    end_date = serializers.DateField(required=False, allow_null=True)
    period = serializers.ChoiceField(choices=PERIOD_CHOICES, default="weekly")
    suburb = serializers.CharField(required=False, allow_blank=True, default="")


class ProfileMatchRequestSerializer(serializers.Serializer):
    crime_type_id = serializers.IntegerField()
    modus_operandi = serializers.CharField(min_length=3)
    time_of_day = serializers.ChoiceField(choices=["morning", "afternoon", "evening", "night"], required=False, allow_blank=True)
    day_of_week = serializers.ChoiceField(
        choices=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        required=False, allow_blank=True
    )
    weapon_used = serializers.CharField(required=False, allow_blank=True, default="")
    top_n = serializers.IntegerField(required=False, default=5, min_value=1, max_value=20)
