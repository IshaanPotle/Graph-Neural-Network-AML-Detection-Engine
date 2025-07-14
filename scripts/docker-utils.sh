#!/bin/bash

# AML Engine Docker Utilities
set -e

show_help() {
    echo "AML Engine Docker Utilities"
    echo "=========================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  logs      - Show logs (all services)"
    echo "  logs-api  - Show API logs only"
    echo "  logs-dash - Show dashboard logs only"
    echo "  status    - Show service status"
    echo "  shell     - Open shell in API container"
    echo "  test      - Run health checks"
    echo "  clean     - Remove containers and images"
    echo "  update    - Rebuild and restart services"
    echo "  help      - Show this help"
    echo ""
}

start_services() {
    echo "🚀 Starting AML Engine services..."
    docker-compose up -d
    echo "✅ Services started!"
}

stop_services() {
    echo "🛑 Stopping AML Engine services..."
    docker-compose down
    echo "✅ Services stopped!"
}

restart_services() {
    echo "🔄 Restarting AML Engine services..."
    docker-compose restart
    echo "✅ Services restarted!"
}

show_logs() {
    echo "📋 Showing logs for all services..."
    docker-compose logs -f
}

show_api_logs() {
    echo "📋 Showing API logs..."
    docker-compose logs -f aml-api
}

show_dashboard_logs() {
    echo "📋 Showing dashboard logs..."
    docker-compose logs -f aml-dashboard
}

show_status() {
    echo "📊 Service Status:"
    echo "=================="
    docker-compose ps
    echo ""
    echo "🔍 Health Checks:"
    echo "================="
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API: Healthy"
    else
        echo "❌ API: Unhealthy"
    fi
    
    # Check Dashboard
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "✅ Dashboard: Running"
    else
        echo "❌ Dashboard: Not responding"
    fi
}

open_shell() {
    echo "🐚 Opening shell in API container..."
    docker-compose exec aml-api /bin/bash
}

run_tests() {
    echo "🧪 Running health checks..."
    
    # Test API endpoints
    echo "Testing API endpoints..."
    curl -s http://localhost:8000/health | jq .status
    curl -s http://localhost:8000/ping | jq .status
    curl -s -H "Authorization: Bearer test-token" http://localhost:8000/dataset/info | jq .dataset_name
    
    # Test dashboard
    echo "Testing dashboard..."
    if curl -s http://localhost:8501 | grep -q "Streamlit"; then
        echo "✅ Dashboard is responding"
    else
        echo "❌ Dashboard is not responding"
    fi
}

clean_up() {
    echo "🧹 Cleaning up Docker resources..."
    read -p "This will remove all containers and images. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down --rmi all --volumes --remove-orphans
        echo "✅ Cleanup completed!"
    else
        echo "❌ Cleanup cancelled."
    fi
}

update_services() {
    echo "🔄 Updating services..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    echo "✅ Services updated!"
}

# Main script logic
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    logs-api)
        show_api_logs
        ;;
    logs-dash)
        show_dashboard_logs
        ;;
    status)
        show_status
        ;;
    shell)
        open_shell
        ;;
    test)
        run_tests
        ;;
    clean)
        clean_up
        ;;
    update)
        update_services
        ;;
    help|*)
        show_help
        ;;
esac 