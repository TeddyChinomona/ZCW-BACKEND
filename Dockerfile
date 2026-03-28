# 1. Use an official lightweight Python image
FROM python:3.14.3

# 2. Set environment variables
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is logged directly to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Install system dependencies (often needed for ML libraries, DB drivers, etc.)
RUN apt-get update && apt-get install -y gdal-bin\
    libgdal-dev \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6. Copy the rest of the Django project
COPY . /app/

# 7. Expose the standard Django port
EXPOSE 8000

# 8. Start the Django development server (Swap to Gunicorn for production later)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]