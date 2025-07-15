# Use a lightweight Nginx image
FROM nginx:alpine

WORKDIR /app

# Remove default static files
RUN rm -rf /usr/share/nginx/html/*

# Copy your static files into Nginx's web directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80
# Start Nginx server