# 🐳 Docker Setup Guide for Book Recommender System

## 📋 Prerequisites

1. **Docker Account Setup:**
   - Create account at [hub.docker.com](https://hub.docker.com)
   - Note your DockerHub username
   - Generate Access Token: Account Settings → Security → New Access Token

2. **GitHub Secrets Setup:**
   Go to your GitHub repository → Settings → Secrets and variables → Actions

   Add these secrets:
   ```
   DOCKERHUB_USERNAME = your_dockerhub_username
   DOCKERHUB_TOKEN = your_dockerhub_access_token
   SLACK_WEBHOOK_URL = your_slack_webhook (optional)
   ```

## 🚀 Quick Start

### Local Development
```bash
# Build and run locally
docker build -t book-recommender .
docker run -p 8000:8000 book-recommender

# Or use Docker Compose
docker-compose up --build
```

### Production Deployment
```bash
# Pull from DockerHub
docker pull your_username/book-recommender-system:latest

# Run container
docker run -d \
  --name book-recommender \
  -p 8000:8000 \
  --restart unless-stopped \
  your_username/book-recommender-system:latest
```

## 🔧 Configuration

### Environment Variables
```bash
PYTHONPATH=/app
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Volume Mounts
```bash
# For persistent data
-v ./data:/app/data

# For configuration
-v ./config:/app/config
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint
```bash
# Check container health
docker ps
docker logs book-recommender

# Manual health check
curl http://localhost:8000/health
```

### Container Stats
```bash
# Monitor resource usage
docker stats book-recommender

# View logs
docker logs -f book-recommender
```

## 🔄 CI/CD Pipeline Features

### Automated Workflow
- ✅ **Code Testing** - Pytest with coverage
- ✅ **Security Scanning** - Trivy vulnerability scanner
- ✅ **Multi-platform Build** - AMD64 + ARM64
- ✅ **DockerHub Push** - Automatic image publishing
- ✅ **Deployment** - Production deployment
- ✅ **Notifications** - Slack integration
- ✅ **Cleanup** - Old image removal

### Trigger Events
- Push to `main` or `develop` branch
- Pull requests to `main`
- Release creation
- Manual workflow dispatch

## 🏷️ Image Tags

### Automatic Tagging
- `latest` - Latest main branch
- `v1.0.0` - Semantic version tags
- `main` - Main branch builds
- `develop` - Development builds
- `pr-123` - Pull request builds

### Manual Tagging
```bash
# Tag specific version
docker tag book-recommender your_username/book-recommender-system:v1.0.0
docker push your_username/book-recommender-system:v1.0.0
```

## 🔒 Security Best Practices

### Image Security
- ✅ Non-root user execution
- ✅ Minimal base image (Python slim)
- ✅ Vulnerability scanning
- ✅ Multi-stage builds
- ✅ Secret management

### Runtime Security
```bash
# Run with security options
docker run \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp \
  -p 8000:8000 \
  book-recommender
```

## 🐛 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   docker build --no-cache -t book-recommender .
   
   # Debug build process
   docker build --progress=plain -t book-recommender .
   ```

2. **Runtime Errors**
   ```bash
   # Check container logs
   docker logs book-recommender
   
   # Interactive debugging
   docker run -it book-recommender /bin/bash
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   
   # Run with user mapping
   docker run --user $(id -u):$(id -g) book-recommender
   ```

### Performance Optimization

1. **Multi-stage Builds**
   ```dockerfile
   # Build stage
   FROM python:3.10 as builder
   # ... build dependencies
   
   # Runtime stage
   FROM python:3.10-slim
   # ... copy artifacts
   ```

2. **Layer Caching**
   ```bash
   # Use BuildKit for better caching
   DOCKER_BUILDKIT=1 docker build .
   ```

## 📈 Scaling & Production

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  book-recommender:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### Load Balancing
```yaml
# Add Nginx load balancer
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

### Monitoring Stack
```yaml
# Add monitoring services
grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
```

## 🚀 Deployment Strategies

### Blue-Green Deployment
```bash
# Deploy new version
docker run -d --name book-recommender-green your_username/book-recommender-system:v2.0.0

# Switch traffic
# Update load balancer configuration

# Remove old version
docker stop book-recommender-blue
docker rm book-recommender-blue
```

### Rolling Updates
```bash
# Using Docker Swarm
docker service update \
  --image your_username/book-recommender-system:v2.0.0 \
  book-recommender-service
```

## 📞 Support

### Getting Help
- 📖 [Docker Documentation](https://docs.docker.com)
- 🐙 [GitHub Actions Docs](https://docs.github.com/en/actions)
- 🐳 [DockerHub](https://hub.docker.com)

### Useful Commands
```bash
# Container management
docker ps -a                    # List all containers
docker images                   # List all images
docker system prune -a          # Clean up everything

# Debugging
docker exec -it container_name /bin/bash  # Enter container
docker inspect container_name             # Container details
docker logs --tail 50 container_name      # Recent logs
```