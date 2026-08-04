#!/usr/bin/env bash
# ============================================
# DeepMemory one-click start/stop
#
#   ./scripts/start-all.sh            # start backend (systemd)
#   ./scripts/start-all.sh --stop     # stop
#   ./scripts/start-all.sh --status   # show status
#   ./scripts/start-all.sh --build    # rebuild frontend
#
# Architecture:
#   Nginx (80/443) -> static files /work/mem/web/dist + reverse proxy :8000
#   Backend: systemd deepmem.service (8000)
#   Frontend: pure static, served directly by nginx, no extra process
# ============================================
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    --stop)
        echo "Stopping backend..."
        sudo systemctl stop deepmem 2>/dev/null && echo "  stopped" || echo "  not running"
        ;;

    --status)
        sudo systemctl status deepmem --no-pager 2>/dev/null | head -5 || echo "backend: not running"
        echo ""
        echo "Backend:  $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 'down')"
        echo "Domain:   $(curl -sk -o /dev/null -w '%{http_code}' https://localhost/health -H 'Host: deepmem.dev' 2>/dev/null || echo 'down')"
        ;;

    --build)
        echo "Building frontend..."
        cd web && npm run build
        echo "  done -> web/dist/"
        ;;

    *)
        sudo systemctl daemon-reload 2>/dev/null
        sudo systemctl enable deepmem 2>/dev/null
        sudo systemctl restart deepmem 2>/dev/null &
        sleep 2

        echo "=== DeepMemory ==="
        echo "  Backend  -> https://deepmem.dev/health"
        echo "  Frontend -> https://deepmem.dev/"
        echo "  Logs     -> logs/deepmem.log"
        echo "  Stop     -> ./scripts/start-all.sh --stop"
        echo "  Build    -> ./scripts/start-all.sh --build"
        sudo systemctl status deepmem --no-pager 2>/dev/null | head -4
        ;;
esac
