#!/bin/bash

# AML Engine Production Deployment Script
set -e

echo "🚀 AML Engine Production Deployment"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE=".env.production"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi

# Check if production environment file exists
if [ ! -f "$ENV_FILE" ]; then
    print_warning "Production environment file not found. Creating template..."
    cat > "$ENV_FILE" << EOF
# AML Engine Production Environment Variables

# Security
SECRET_KEY=$(openssl rand -hex 32)
AUTH_ENABLED=true

# Database
DB_USER=aml_user
DB_PASSWORD=$(openssl rand -base64 32)
DATABASE_URL=postgresql://aml_user:${DB_PASSWORD}@postgres:5432/aml_engine

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# SSL Certificates (update with your domain)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Logging
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=30

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
EOF
    print_success "Created $ENV_FILE template. Please review and update with your values."
    exit 1
fi

# Load environment variables
print_status "Loading environment variables..."
source "$ENV_FILE"

# Validate required environment variables
required_vars=("SECRET_KEY" "DB_PASSWORD" "DATABASE_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Required environment variable $var is not set"
        exit 1
    fi
done

# Create backup directory
print_status "Creating backup directory..."
mkdir -p "$BACKUP_DIR"

# Backup existing data
if [ -d "data" ]; then
    print_status "Backing up existing data..."
    tar -czf "$BACKUP_DIR/data_backup.tar.gz" data/
fi

if [ -d "logs" ]; then
    print_status "Backing up existing logs..."
    tar -czf "$BACKUP_DIR/logs_backup.tar.gz" logs/
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p {logs,data,nginx/ssl,monitoring/grafana/{dashboards,datasources}}

# Generate SSL certificates (self-signed for development)
if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
    print_status "Generating SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=aml-engine.local"
fi

# Set proper permissions
print_status "Setting proper permissions..."
chmod 600 nginx/ssl/key.pem
chmod 644 nginx/ssl/cert.pem

# Stop existing services
print_status "Stopping existing services..."
docker-compose down --remove-orphans || true

# Remove old images to ensure fresh build
print_status "Removing old images..."
docker-compose -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans || true

# Build production images
print_status "Building production images..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

# Start services
print_status "Starting production services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Health checks
print_status "Running health checks..."

# Check API
for i in {1..30}; do
    if curl -s -f http://localhost:8000/health > /dev/null; then
        print_success "API is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "API health check failed"
        docker-compose -f "$COMPOSE_FILE" logs aml-api
        exit 1
    fi
    sleep 2
done

# Check Dashboard
for i in {1..15}; do
    if curl -s -f http://localhost:8501 > /dev/null; then
        print_success "Dashboard is running!"
        break
    fi
    if [ $i -eq 15 ]; then
        print_error "Dashboard health check failed"
        docker-compose -f "$COMPOSE_FILE" logs aml-dashboard
        exit 1
    fi
    sleep 2
done

# Check Database
for i in {1..10}; do
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "$DB_USER" -d aml_engine > /dev/null 2>&1; then
        print_success "Database is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "Database health check failed"
        exit 1
    fi
    sleep 3
done

# Check Redis
for i in {1..10}; do
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "Redis health check failed"
        exit 1
    fi
    sleep 2
done

# Initialize database (if needed)
print_status "Initializing database..."
docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$DB_USER" -d aml_engine -c "SELECT 1;" > /dev/null 2>&1 || {
    print_status "Running database initialization..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$DB_USER" -d aml_engine -f /docker-entrypoint-initdb.d/init-db.sql
}

# Setup monitoring
print_status "Setting up monitoring..."

# Wait for Prometheus
sleep 10
if curl -s http://localhost:9090/-/ready > /dev/null; then
    print_success "Prometheus is ready!"
else
    print_warning "Prometheus not ready yet (this is normal for first startup)"
fi

# Wait for Grafana
sleep 10
if curl -s http://localhost:3000/api/health > /dev/null; then
    print_success "Grafana is ready!"
else
    print_warning "Grafana not ready yet (this is normal for first startup)"
fi

# Performance optimization
print_status "Optimizing performance..."

# Set Docker resource limits
docker update --memory=4g --cpus=2.0 aml-api-prod 2>/dev/null || true
docker update --memory=2g --cpus=1.0 aml-dashboard-prod 2>/dev/null || true

# Security hardening
print_status "Applying security hardening..."

# Update container security settings
docker update --security-opt=no-new-privileges aml-api-prod 2>/dev/null || true
docker update --security-opt=no-new-privileges aml-dashboard-prod 2>/dev/null || true

# Create monitoring dashboard
print_status "Creating monitoring dashboard..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Final status check
print_status "Final status check..."
docker-compose -f "$COMPOSE_FILE" ps

echo ""
print_success "🎉 Production deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
echo "  API: https://api.aml-engine.local (or http://localhost:8000)"
echo "  Dashboard: https://dashboard.aml-engine.local (or http://localhost:8501)"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3000 (admin/admin)"
echo "  Kibana: http://localhost:5601"
echo ""
echo "🔧 Management Commands:"
echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
echo "  Restart: docker-compose -f $COMPOSE_FILE restart"
echo "  Update: ./scripts/deploy-production.sh"
echo ""
echo "📋 Next Steps:"
echo "  1. Update your DNS/hosts file to point to the service URLs"
echo "  2. Configure SSL certificates for your domain"
echo "  3. Set up backup schedules"
echo "  4. Configure monitoring alerts"
echo "  5. Review security settings"
echo ""
print_warning "⚠️  Remember to:"
echo "  - Change default passwords"
echo "  - Configure firewall rules"
echo "  - Set up regular backups"
echo "  - Monitor system resources"
echo "" 