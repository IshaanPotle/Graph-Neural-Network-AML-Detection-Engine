# AML Engine - Docker Setup

This document explains how to run the AML Engine using Docker for consistent, reproducible deployments.

## 🐳 Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed
- Data files in `data/elliptic_raw/` directory

### One-Command Setup
```bash
./scripts/docker-setup.sh
```

This script will:
- Check prerequisites
- Build Docker images
- Start all services
- Verify health checks
- Display access URLs

## 📋 Manual Setup

### 1. Build and Start Services
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d
```

### 2. Check Status
```bash
# View running services
docker-compose ps

# Check health
./scripts/docker-utils.sh status
```

### 3. Access Services
- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## 🔧 Docker Utilities

Use the utility script for common operations:

```bash
# Show all available commands
./scripts/docker-utils.sh help

# Start services
./scripts/docker-utils.sh start

# Stop services
./scripts/docker-utils.sh stop

# View logs
./scripts/docker-utils.sh logs

# Check status
./scripts/docker-utils.sh status

# Run health tests
./scripts/docker-utils.sh test

# Open shell in container
./scripts/docker-utils.sh shell
```

## 🏗️ Architecture

### Services
1. **aml-api** - FastAPI backend (port 8000)
2. **aml-dashboard** - Streamlit frontend (port 8501)
3. **aml-network** - Internal Docker network

### Volumes
- `./data` → `/app/data` - Data files
- `./logs` → `/app/logs` - Application logs

### Environment Variables
- `GNN_FRAMEWORK=pyg` - Graph framework
- `PYTHONPATH=/app` - Python path
- `LOG_LEVEL=INFO` - Logging level

## 📊 Monitoring

### Health Checks
- API health check: `GET /health`
- Automatic restart on failure
- 30-second check intervals

### Logs
```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f aml-api

# Dashboard only
docker-compose logs -f aml-dashboard
```

## 🔄 Updates

### Rebuild and Restart
```bash
./scripts/docker-utils.sh update
```

### Manual Update
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🧹 Cleanup

### Remove Everything
```bash
./scripts/docker-utils.sh clean
```

### Manual Cleanup
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Remove volumes
docker-compose down --volumes

# Remove everything
docker-compose down --rmi all --volumes --remove-orphans
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   lsof -i :8501
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Permission Issues**
   ```bash
   # Fix script permissions
   chmod +x scripts/*.sh
   ```

3. **Data Files Missing**
   ```bash
   # Check data directory
   ls -la data/elliptic_raw/
   
   # Ensure files exist:
   # - wallets_features.csv
   # - AddrAddr_edgelist.csv
   # - wallets_classes.csv
   ```

4. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs aml-api
   docker-compose logs aml-dashboard
   
   # Check resource usage
   docker stats
   ```

### Debug Mode
```bash
# Run in foreground with logs
docker-compose up

# Run single service
docker-compose up aml-api
```

## 🔒 Security

### Production Considerations
1. **Authentication**: Implement proper auth tokens
2. **HTTPS**: Use reverse proxy with SSL
3. **Network**: Restrict container network access
4. **Secrets**: Use Docker secrets for sensitive data
5. **Resource Limits**: Set memory/CPU limits

### Example Production Setup
```yaml
# Add to docker-compose.yml
services:
  aml-api:
    environment:
      - AUTH_ENABLED=true
      - SECRET_KEY=${SECRET_KEY}
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale API service
docker-compose up -d --scale aml-api=3

# Use load balancer
docker-compose up -d nginx
```

### Performance Tuning
- Increase memory limits for large datasets
- Use GPU containers for inference
- Add Redis for caching
- Use volume mounts for persistent data

## 🔗 Integration

### External Services
- **Redis**: Uncomment in docker-compose.yml
- **PostgreSQL**: Add database service
- **Nginx**: Add reverse proxy
- **Prometheus**: Add monitoring

### CI/CD
```yaml
# Example GitHub Actions
- name: Build and Deploy
  run: |
    docker-compose build
    docker-compose up -d
    ./scripts/docker-utils.sh test
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/deploy/deploy-streamlit-using-docker) 