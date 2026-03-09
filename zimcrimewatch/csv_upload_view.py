"""
CSVUploadView — add this class to the bottom of zimcrimewatch/views.py
=======================================================================
POST /api/zrp/data/upload-csv/

Accepts a multipart/form-data request with one field named `file`
containing a .csv that matches the zimcrime_watch_dataset schema.

Access: Admin only (IsZRPAdmin permission).

Response (200 OK):
  { "created": N, "skipped": N, "errors": [ {row, case_number, reason} ] }
"""
import io
import pandas as pd
from django.contrib.gis.geos import Point
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CrimeIncident, CrimeType
from .permissions import IsZRPAdmin

# ---------------------------------------------------------------------------
# Required CSV columns validated before any DB write is attempted
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "case_number", "crime_type", "timestamp",
    "latitude", "longitude", "modus_operandi", "status",
}

# Map CSV status strings → CrimeIncident.STATUS_CHOICES keys
STATUS_MAP = {
    "reported":            "reported",
    "under investigation": "under_investigation",
    "under_investigation": "under_investigation",
    "closed":              "closed",
    "unsolved":            "unsolved",
}

# Map CSV time_of_day strings → model choices
TIME_OF_DAY_MAP = {
    "morning":   "morning",
    "afternoon": "afternoon",
    "evening":   "evening",
    "night":     "night",
}


class CSVUploadView(APIView):
    """
    POST /api/zrp/data/upload-csv/

    Workflow:
      1. Validate file is present and is a .csv
      2. Parse CSV into a pandas DataFrame
      3. Confirm all required columns exist
      4. Load existing case_numbers into a set for fast O(1) duplicate checks
      5. Cache all CrimeType objects to avoid per-row DB queries
      6. Iterate rows — resolve FK, parse coords/timestamp, build instances
      7. bulk_create all valid rows in one DB round-trip (batch_size=500)
      8. Return a summary { created, skipped, errors }
    """

    # Tell DRF to parse multipart bodies so request.FILES is populated
    parser_classes = [MultiPartParser]

    # Only ZRP admins may bulk-upload data (fulfils ADM-03)
    permission_classes = [IsZRPAdmin]

    def post(self, request):
        # ── 1. Pull the uploaded file ──────────────────────────────────
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response(
                {"detail": "No file provided. Send the CSV under the key 'file'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not uploaded_file.name.lower().endswith(".csv"):
            return Response(
                {"detail": f"Invalid file type '{uploaded_file.name}'. Only .csv files are accepted."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 2. Parse the CSV into a DataFrame ─────────────────────────
        try:
            # io.StringIO lets pandas read from memory without a temp file
            content = uploaded_file.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
        except Exception as exc:
            return Response(
                {"detail": f"Could not parse CSV: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 3. Validate required columns are present ───────────────────
        missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
        if missing:
            return Response(
                {"detail": f"CSV is missing required columns: {sorted(missing)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Normalise all column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # ── 4. Fetch existing case_numbers into a set ──────────────────
        # A set gives O(1) lookup — far faster than one DB query per row
        existing_case_numbers = set(
            CrimeIncident.objects.values_list("case_number", flat=True)
        )

        # ── 5. Cache CrimeType name → object to avoid repeated DB hits ─
        crime_type_cache: dict[str, CrimeType] = {
            ct.name.lower(): ct for ct in CrimeType.objects.all()
        }

        # ── 6. Build CrimeIncident instances row by row ────────────────
        to_create: list[CrimeIncident] = []
        skipped = 0
        errors: list[dict] = []

        for idx, row in df.iterrows():
            row_num = int(idx) + 2  # +2: 1-indexed + header row

            case_number = str(row.get("case_number", "")).strip()
            if not case_number:
                errors.append({"row": row_num, "case_number": "", "reason": "Empty case_number."})
                continue

            # Skip duplicates already in the database
            if case_number in existing_case_numbers:
                skipped += 1
                continue

            # ── Resolve (or auto-create) the CrimeType FK ──────────────
            crime_type_name = str(row.get("crime_type", "")).strip().lower()
            crime_type_obj = crime_type_cache.get(crime_type_name)
            if crime_type_obj is None:
                # Auto-create unknown crime types so no row is silently lost
                crime_type_obj, _ = CrimeType.objects.get_or_create(
                    name=crime_type_name.title(),
                    defaults={"description": "", "icon": "📍"},
                )
                crime_type_cache[crime_type_name] = crime_type_obj  # update cache

            # ── Parse coordinates ───────────────────────────────────────
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, TypeError):
                errors.append({
                    "row": row_num,
                    "case_number": case_number,
                    "reason": "Invalid latitude/longitude — must be numeric.",
                })
                continue

            # PostGIS Point(x=longitude, y=latitude) — note axis order
            location_point = Point(lon, lat, srid=4326)

            # ── Parse timestamp ─────────────────────────────────────────
            try:
                timestamp = pd.to_datetime(row["timestamp"], utc=True)
            except Exception:
                errors.append({
                    "row": row_num,
                    "case_number": case_number,
                    "reason": f"Cannot parse timestamp '{row.get('timestamp')}'.",
                })
                continue

            # ── Normalise status and time_of_day strings ────────────────
            incident_status = STATUS_MAP.get(
                str(row.get("status", "")).strip().lower(), "reported"
            )
            time_of_day = TIME_OF_DAY_MAP.get(
                str(row.get("time_of_day", "")).strip().lower(), "morning"
            )

            # ── Build the unsaved model instance ────────────────────────
            to_create.append(CrimeIncident(
                case_number=case_number,
                crime_type=crime_type_obj,
                timestamp=timestamp,
                location=location_point,
                modus_operandi=str(row.get("modus_operandi", "")).strip(),
                status=incident_status,
                time_of_day=time_of_day,
                # pandas dayofweek: 0=Monday … 6=Sunday, matches Python convention
                day_of_week=int(timestamp.dayofweek),
            ))

            # Track the new case_number in memory to catch intra-CSV duplicates
            # without an extra DB round-trip
            existing_case_numbers.add(case_number)

        # ── 7. Single bulk INSERT for all valid rows ───────────────────
        # bulk_create issues one SQL statement instead of N individual INSERTs.
        # batch_size=500 prevents the query from becoming too large for the DB.
        if to_create:
            CrimeIncident.objects.bulk_create(to_create, batch_size=500)

        # ── 8. Return upload summary ───────────────────────────────────
        return Response(
            {
                "created": len(to_create),
                "skipped": skipped,
                "errors":  errors,
            },
            status=status.HTTP_200_OK,
        )