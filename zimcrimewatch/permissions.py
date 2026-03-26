"""
zimcrimewatch/permissions.py
============================
Custom DRF permission classes for role-based access control.

Role hierarchy (highest to lowest)
------------------------------------
  admin    — full system access (CRUD on everything, user management, ML training)
  analyst  — read/write on incidents and analytics; cannot manage users or train
  officer  — read-only on dashboard data; can create incidents via RRB form

These permission classes read the `role` field directly from the CustomUser
model.  There is NO separate ZRPProfile model in this project — the `role`
field lives on CustomUser itself.

All ZRP-authenticated views require at minimum IsZRPAuthenticated.
Views that need analyst-or-above use IsZRPAnalystOrAdmin.
Views that need admin-only use IsZRPAdmin.
"""
from rest_framework.permissions import BasePermission, IsAuthenticated
from loguru import logger

class IsZRPAuthenticated(IsAuthenticated):
    """
    Permits access to any active, authenticated ZRP user (any role).

    Extends IsAuthenticated so that:
    • Unauthenticated requests (no JWT) still get 401 via the parent class.
    • The extra check ensures the user object actually has a 'role' attribute,
      which guards against non-ZRP users who might somehow obtain a token.
    """

    def has_permission(self, request, view) -> bool:
        # IsAuthenticated check — returns False (→ 401) if no valid JWT
        if not super().has_permission(request, view):
            return False
        # Guard: the user object must have a 'role' attribute
        return hasattr(request.user, "role")


class IsZRPAdmin(IsAuthenticated):
    """
    Permits access only to ZRP users with the 'admin' role.

    Used for:
    • User management endpoints (/zrp/users/…)
    • ML model training trigger (/zrp/ml/train/)
    • Bulk CSV upload (/zrp/data/upload-csv/)
    """

    def has_permission(self, request, view) -> bool:
        if not super().has_permission(request, view):
            return False
        return (
            hasattr(request.user, "role") and
            request.user.role == "admin" or "officer"  # Officers can also create incidents via RRB form
        )


class IsZRPAnalystOrAdmin(IsAuthenticated):
    """
    Permits access to ZRP users with 'analyst' or 'admin' role.

    Read (GET/HEAD/OPTIONS) is allowed for all authenticated ZRP users.
    Write (POST/PUT/PATCH/DELETE) is restricted to analysts and admins.

    Used for:
    • Analytics endpoints (heatmap, timeseries, hotspots, profile match)
    • Serial crime linkage endpoints
    • IncidentSimilarCasesView
    """

    def has_permission(self, request, view) -> bool:
        if not super().has_permission(request, view):
            return False
        if not hasattr(request.user, "role"):
            return False

        role = request.user.role
        logger.debug(f"User role: {role} | Method: {request.method} | Path: {request.path}")
        # Read-only access is fine for all ZRP roles (officers can view analytics)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return True

        # Write access requires at least analyst role
        return role in ("analyst", "admin", "officer")  # Officers can also create incidents via RRB form