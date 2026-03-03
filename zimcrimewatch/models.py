"""
ZimCrimeWatch - Django Models
Defines all database entities for the ZRP crime analytics system.
"""
from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.contrib.gis.db import models as gis_models
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin
)

class CustomUserManager(BaseUserManager):
    """Manager for CustomUser model"""
    def create_superuser(self, username, first_name, last_name, zrp_badge_number, password, **other_fields):
        """Create and save a superuser with the given details."""
        other_fields.setdefault('is_superuser', True)
        other_fields.setdefault('is_staff', True)
        other_fields.setdefault('is_active', True)

        if other_fields.get('is_superuser') is not True:
            raise ValidationError(message= "Superuser must be set to True")

        if other_fields.get('is_staff') is not True:
            raise ValidationError(message= "Staff must be set to True")

        if other_fields.get('is_active') is not True:
            raise ValidationError(message= "Active must be set to True")

        return self.create_user(username, first_name, last_name, zrp_badge_number, password, **other_fields)

    def create_user(self, username, first_name, last_name, zrp_badge_number, password, **other_fields):
        """Create and save a regular user with the given details."""
        if not zrp_badge_number:
            raise ValidationError(message="ZRP badge number cannot be empty")

        user = self.model(
            fullname = f"{first_name} {last_name}",
            username=username,
            first_name=first_name,
            last_name=last_name,
            zrp_badge_number=zrp_badge_number,
            **other_fields
        )
        user.set_password(password)
        user.save()
        return user

class CustomUser(AbstractBaseUser, PermissionsMixin):
    """Custom User model"""
    ROLE_CHOICES = [
        ("analyst", "Analyst"),
        ("officer", "Officer"),
        ("admin", "Administrator"),
    ]
    fullname = models.CharField(max_length=255, null=True)
    username = models.CharField(max_length=20)
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)
    zrp_badge_number = models.CharField(max_length=20, unique=True)
    base_station = models.ForeignKey('BaseStation', on_delete=models.PROTECT, null=True, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='PUBLIC')
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=True)

    objects = CustomUserManager()
    USERNAME_FIELD = 'zrp_badge_number'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']
    
    def __str__(self):
        return f"Username: {self.username}  Active: {self.is_active}"

class Station(models.Model):
    """Model representing a police station."""
    code = models.CharField(max_length=10, unique=True, null=True)
    name = models.CharField(max_length=100)
    location = gis_models.PointField(srid=4326, null=True, blank=True)

    def __str__(self):
        return f"Station: {self.name}"

class BaseStation(models.Model):
    """Model representing a base station."""
    name = models.CharField(max_length=100)
    station = models.ForeignKey(
        'Station',
        on_delete=models.PROTECT,
        related_name='base_stations',
        blank=True
    )
    location = gis_models.PointField(srid=4326, null=True, blank=True)

    def __str__(self):
        return f"Base Station: {self.name}"

# class ZRPProfile(models.Model):
#     """Extends the base Django User with ZRP-specific role information."""
    
#     full_name = models.CharField(max_length=255)
#     badge_number = models.CharField(max_length=50, blank=True, default="")
#     station = models.CharField(max_length=150, blank=True, default="")

#     def __str__(self):
#         return f"{self.full_name} ({self.role})"

class CrimeType(models.Model):
    """Lookup table for crime categories used across the system."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")
    icon = models.CharField(max_length=50, blank=True, default="")  # icon slug for frontend

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

class CrimeIncident(models.Model):
    """Core entity representing a single crime incident reported to the ZRP."""
    STATUS_CHOICES = [
        ("reported", "Reported"),
        ("under_investigation", "Under Investigation"),
        ("closed", "Closed"),
        ("unsolved", "Unsolved"),
    ]
    TIME_OF_DAY_CHOICES = [
        ("morning", "Morning (06:00–12:00)"),
        ("afternoon", "Afternoon (12:00–18:00)"),
        ("evening", "Evening (18:00–22:00)"),
        ("night", "Night (22:00–06:00)"),
    ]

    case_number = models.CharField(max_length=50, unique=True)
    crime_type = models.ForeignKey(CrimeType, on_delete=models.PROTECT, related_name="incidents")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, db_index=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, db_index=True)
    suburb = models.CharField(max_length=150, blank=True, default="")
    description_narrative = models.TextField(blank=True, default="")
    modus_operandi = models.TextField(blank=True, default="")
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="reported", db_index=True)
    weapon_used = models.CharField(max_length=100, blank=True, default="")
    num_suspects = models.PositiveIntegerField(default=0)
    # Derived/cached fields for ML feature engineering
    time_of_day = models.CharField(max_length=20, choices=TIME_OF_DAY_CHOICES, blank=True, default="")
    day_of_week = models.CharField(max_length=10, blank=True, default="")
    # Linkage field for serial crimes (analyst manually assigns a group label)
    serial_group_label = models.CharField(max_length=100, blank=True, default="",help_text="Group label linking serial cases, e.g. 'Mbare Burglar 2024'")
    created_by = models.ForeignKey(CustomUser, null=True, blank=True, on_delete=models.SET_NULL,related_name="incidents_created")
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["crime_type", "timestamp"]),
            models.Index(fields=["status", "timestamp"]),
        ]

    def save(self, *args, **kwargs):
        """Auto-derive time_of_day and day_of_week before saving."""
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
        return f"{self.case_number} – {self.crime_type.name} ({self.timestamp.date()})"
