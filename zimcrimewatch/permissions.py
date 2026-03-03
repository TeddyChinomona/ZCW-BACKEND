"""
ZimCrimeWatch - Custom DRF Permissions
Role-based access control for ZRP officers vs public.
"""
from rest_framework.permissions import BasePermission, IsAuthenticated


class IsZRPAuthenticated(IsAuthenticated):
    """Allows access only to authenticated ZRP users (any role)."""

    def has_permission(self, request, view):
        return (
            super().has_permission(request, view)
            and hasattr(request.user, "zrp_profile")
        )


class IsZRPAdmin(IsAuthenticated):
    """Allows access only to ZRP users with the 'admin' role."""

    def has_permission(self, request, view):
        return (
            super().has_permission(request, view)
            and hasattr(request.user, "zrp_profile")
            and request.user.zrp_profile.role == "admin"
        )


class IsZRPAnalystOrAdmin(IsAuthenticated):
    """Allows read/write for analysts and admins; officers are read-only."""

    def has_permission(self, request, view):
        if not super().has_permission(request, view):
            return False
        if not hasattr(request.user, "zrp_profile"):
            return False
        role = request.user.zrp_profile.role
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return True  # all ZRP roles can read
        return role in ("analyst", "admin")
