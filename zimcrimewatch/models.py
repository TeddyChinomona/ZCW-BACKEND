"""
ZimCrimeWatch - Django Models
Defines all database entities for the ZRP crime analytics system.
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class CrimeType(models.Model):
    """Lookup table for crime categories used across the system."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")
    icon = models.CharField(max_length=50, blank=True, default="")  # icon slug for frontend

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class ZRPProfile(models.Model):
    """Extends the base Django User with ZRP-specific role information."""
    ROLE_CHOICES = [
        ("analyst", "Analyst"),
        ("officer", "Officer"),
        ("admin", "Administrator"),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="zrp_profile")
    full_name = models.CharField(max_length=255)
    badge_number = models.CharField(max_length=50, blank=True, default="")
    station = models.CharField(max_length=150, blank=True, default="")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="analyst")

    def __str__(self):
        return f"{self.full_name} ({self.role})"


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
    serial_group_label = models.CharField(max_length=100, blank=True, default="",
                                          help_text="Group label linking serial cases, e.g. 'Mbare Burglar 2024'")
    created_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="incidents_created")
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
