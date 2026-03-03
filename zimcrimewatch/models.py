"""
ZimCrimeWatch - Django Models
Defines all database entities for the ZRP crime analytics system.

PostGIS Note
------------
Longitude / latitude are no longer stored as separate DecimalFields.
They are derived at runtime from the single `location` PointField:
  location.x  → longitude
  location.y  → latitude
This gives full spatial-query support (distance, bbox, containment, etc.)
without redundant columns.
"""
from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.contrib.gis.db import models as gis_models
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)


# =============================================================================
# User management
# =============================================================================

class CustomUserManager(BaseUserManager):
    """Manager for CustomUser model."""

    def create_user(
        self, username, first_name, last_name, zrp_badge_number, password, **extra_fields
    ):
        if not zrp_badge_number:
            raise ValidationError("ZRP badge number cannot be empty.")
        user = self.model(
            fullname=f"{first_name} {last_name}",
            username=username,
            first_name=first_name,
            last_name=last_name,
            zrp_badge_number=zrp_badge_number,
            **extra_fields,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(
        self, username, first_name, last_name, zrp_badge_number, password, **extra_fields
    ):
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_active", True)

        for flag in ("is_superuser", "is_staff", "is_active"):
            if extra_fields.get(flag) is not True:
                raise ValidationError(f"{flag} must be True for superuser.")

        return self.create_user(
            username, first_name, last_name, zrp_badge_number, password, **extra_fields
        )


class CustomUser(AbstractBaseUser, PermissionsMixin):
    """
    Custom User model.
    USERNAME_FIELD = 'zrp_badge_number'  — used for authentication.
    """

    ROLE_CHOICES = [
        ("analyst", "Analyst"),
        ("officer", "Officer"),
        ("admin",   "Administrator"),
    ]

    fullname          = models.CharField(max_length=255, null=True, blank=True)
    username          = models.CharField(max_length=20, unique=True)
    first_name        = models.CharField(max_length=20)
    last_name         = models.CharField(max_length=20)
    zrp_badge_number  = models.CharField(max_length=20, unique=True)
    base_station      = models.ForeignKey(
        "BaseStation", on_delete=models.PROTECT, null=True, blank=True
    )
    role              = models.CharField(max_length=10, choices=ROLE_CHOICES, default="officer")
    is_active         = models.BooleanField(default=True)
    is_staff          = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD  = "zrp_badge_number"
    REQUIRED_FIELDS = ["username", "first_name", "last_name"]

    def __str__(self):
        return f"{self.username} [{self.zrp_badge_number}]"


# =============================================================================
# Station models
# =============================================================================

class Station(models.Model):
    """A top-level ZRP police station."""

    code     = models.CharField(max_length=10, unique=True, null=True, blank=True)
    name     = models.CharField(max_length=100)
    location = gis_models.PointField(srid=4326, null=True, blank=True)

    def __str__(self):
        return f"Station: {self.name}"


class BaseStation(models.Model):
    """A sub-station / patrol base attached to a Station."""

    name     = models.CharField(max_length=100)
    station  = models.ForeignKey(
        "Station", on_delete=models.PROTECT, related_name="base_stations"
    )
    location = gis_models.PointField(srid=4326, null=True, blank=True)

    def __str__(self):
        return f"Base Station: {self.name}"


# =============================================================================
# Crime lookups
# =============================================================================

class CrimeType(models.Model):
    """Lookup table for crime categories."""

    name        = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")
    icon        = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


# =============================================================================
# Crime Incident  (core entity)
# =============================================================================

class CrimeIncident(models.Model):
    """
    A single crime incident reported to the ZRP.

    Spatial storage
    ---------------
    Coordinates are held in a single PostGIS PointField (SRID 4326,
    i.e. WGS-84 / GPS coordinates).

    The `latitude` and `longitude` Python properties give backward-compatible
    access so the rest of the codebase does not need to change.

    The serializer exposes read-only `latitude` / `longitude` fields and
    accepts a `location` write field as either a GeoJSON dict:
        {"type": "Point", "coordinates": [lon, lat]}
    or a plain dict {"longitude": ..., "latitude": ...} via the serializer.
    """

    STATUS_CHOICES = [
        ("reported",           "Reported"),
        ("under_investigation","Under Investigation"),
        ("closed",             "Closed"),
        ("unsolved",           "Unsolved"),
    ]
    TIME_OF_DAY_CHOICES = [
        ("morning",   "Morning (06:00–12:00)"),
        ("afternoon", "Afternoon (12:00–18:00)"),
        ("evening",   "Evening (18:00–22:00)"),
        ("night",     "Night (22:00–06:00)"),
    ]

    case_number  = models.CharField(max_length=50, unique=True)
    crime_type   = models.ForeignKey(
        CrimeType, on_delete=models.PROTECT, related_name="incidents"
    )
    timestamp    = models.DateTimeField(default=timezone.now, db_index=True)

    # ── Primary spatial column ───────────────────────────────────────────────
    location = gis_models.PointField(
        srid=4326,
        null=True,
        blank=True,
        help_text=(
            "WGS-84 point geometry (PostGIS). "
            "Read: location.x = longitude, location.y = latitude. "
            "Write via GeoJSON: {\"type\": \"Point\", \"coordinates\": [lon, lat]}"
        ),
    )
    # ────────────────────────────────────────────────────────────────────────

    suburb                = models.CharField(max_length=150, blank=True, default="")
    description_narrative = models.TextField(blank=True, default="")
    modus_operandi        = models.TextField(blank=True, default="")
    status                = models.CharField(
        max_length=30, choices=STATUS_CHOICES, default="reported", db_index=True
    )
    weapon_used   = models.CharField(max_length=100, blank=True, default="")
    num_suspects  = models.PositiveIntegerField(default=0)

    # Derived / cached fields for ML feature engineering
    time_of_day = models.CharField(
        max_length=20, choices=TIME_OF_DAY_CHOICES, blank=True, default=""
    )
    day_of_week = models.CharField(max_length=10, blank=True, default="")

    # Serial-crime linkage (analyst-assigned)
    serial_group_label = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Group label linking serial cases, e.g. 'Mbare Burglar 2024'",
    )

    created_by = models.ForeignKey(
        CustomUser,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="incidents_created",
    )
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["crime_type", "timestamp"]),
            models.Index(fields=["status",     "timestamp"]),
        ]

    # ------------------------------------------------------------------
    # Backward-compatible properties — keep callers using .latitude / .longitude
    # ------------------------------------------------------------------

    @property
    def latitude(self) -> float | None:
        """WGS-84 latitude (Y coordinate) from the PostGIS PointField."""
        return self.location.y if self.location else None

    @property
    def longitude(self) -> float | None:
        """WGS-84 longitude (X coordinate) from the PostGIS PointField."""
        return self.location.x if self.location else None

    # ------------------------------------------------------------------

    def save(self, *args, **kwargs):
        """Auto-derive time_of_day and day_of_week from timestamp before saving."""
        if self.timestamp:
            hour = self.timestamp.hour
            if 6 <= hour < 12:
                self.time_of_day = "morning"
            elif 12 <= hour < 18:
                self.time_of_day = "afternoon"
            elif 18 <= hour < 22:
                self.time_of_day = "evening"
            else:
                self.time_of_day = "night"
            self.day_of_week = self.timestamp.strftime("%A").lower()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Incident {self.case_number} — {self.crime_type}"
