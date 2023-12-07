# Use the zenmldocker/zenml image as the base image
FROM zenmldocker/zenml:0.50.0

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Install SQLAlchmey
RUN pip install SQLAlchemy==1.4.41 --no-cache-dir