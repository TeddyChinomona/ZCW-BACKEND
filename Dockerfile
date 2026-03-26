# Use a specific version tag; 3.14.3 is the latest stable as of March 2026
FROM python:3.14.3

# Set the working directory inside the container
WORKDIR /app

# Copy all local files into the zimcrimewatch/ directory in the container
COPY ./ zimcrimewatch/

# Change context to the folder where manage.py is located
WORKDIR /app/zimcrimewatch

# Install dependencies (chained to reduce layers)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt

# Use CMD for the final command to start the server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
