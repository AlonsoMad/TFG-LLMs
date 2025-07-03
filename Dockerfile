FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_web.txt .
# Install the required packages
RUN pip install --no-cache-dir -r requirements_web.txt 

# Copy the rest of the application code into the container
COPY ./app ./app
COPY ./backend ./backend 

# Expose the Flask default port
EXPOSE 5000

CMD ["python", "-m", "app.main"]