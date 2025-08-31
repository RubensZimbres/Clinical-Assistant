FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX 1: Copy the saved models into the image ---
# This copies your local "saved_models" folder to "/app/saved_models" inside the container
COPY . .

# --- FIX 2: Copy the application code without the extra nesting ---
# This copies the *contents* of your local "app" folder into "/app"
COPY ./app .

# Expose the port the app runs on
EXPOSE 8080

# --- FIX 3: Update the command to match the new structure ---
# Because main.py is now at /app/main.py, you don't need "app." prefix
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]