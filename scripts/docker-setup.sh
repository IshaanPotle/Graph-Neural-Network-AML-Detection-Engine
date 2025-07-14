#!/bin/bash

# AML Engine Docker Setup Script
set -e

echo "🚀 AML Engine Docker Setup"
echo "=========================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if data files exist
if [ ! -f "data/elliptic_raw/wallets_features.csv" ]; then
    echo "⚠️  Warning: Data files not found in data/elliptic_raw/"
    echo "   Please ensure you have the following files:"
    echo "   - wallets_features.csv"
    echo "   - AddrAddr_edgelist.csv"
    echo "   - wallets_classes.csv"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs

echo "📦 Building Docker images..."
docker-compose build

echo "🔧 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check API health
echo "🏥 Checking API health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ API is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API health check failed after 30 attempts"
        docker-compose logs aml-api
        exit 1
    fi
    sleep 2
done

# Check dashboard
echo "📊 Checking dashboard..."
for i in {1..15}; do
    if curl -s http://localhost:8501 > /dev/null; then
        echo "✅ Dashboard is running!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Dashboard check failed after 15 attempts"
        docker-compose logs aml-dashboard
        exit 1
    fi
    sleep 2
done

echo ""
echo "🎉 AML Engine is now running!"
echo "=============================="
echo "📊 Dashboard: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart: docker-compose restart"
echo "  Update: docker-compose pull && docker-compose up -d"
echo "" 