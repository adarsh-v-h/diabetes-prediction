FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

#Forces stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# To a copy of there requirements
COPY requirements.txt .

# To install all dependencies
# use --no-cache-dir to avoid storing build artifacts
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /

RUN cd /main

# Now the commands to run in sequence
CMD [ "python", "/main.py"]
