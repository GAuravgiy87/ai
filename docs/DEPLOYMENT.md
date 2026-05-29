# 🚀 AI Vigilance - Docker Deployment Guide

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- At least 16GB RAM for full deployment
- 50GB free disk space for recordings
- Linux kernel 5.10+ (for hardware acceleration on Linux)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/ai-vigilance.git
cd ai-vigilance
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your own passwords and settings
nano .env
```

**Important:** Change the default passwords:

- `DB_PASSWORD` - PostgreSQL password
- `REDIS_PASSWORD` - Redis password

### 3. Build Docker Image

```bash
docker-compose build
```

### 4. Start Services

```bash
docker-compose up -d
```

Monitor startup:

```bash
docker-compose logs -f main_app
```

Services startup sequence (automatic):

1. PostgreSQL (database)
2. Redis (cache)
3. Main App (web UI)
4. Camera Server
5. AI Inference Worker
6. Recording Service
7. Analytics Worker
8. Nginx Gateway

### 5. Access Application

- **Web UI**: http://localhost or http://your-server-ip
- **API Docs**: http://localhost/docs
- **Camera Server**: http://localhost:9001 (internal API)

### 6. Verify All Services

```bash
docker-compose ps
```

All containers should show `healthy` or `running` status.

---

## Environment Configuration (.env)

### Database

- `DB_USER` - PostgreSQL username (default: `aiv_user`)
- `DB_PASSWORD` - **CHANGE THIS** - PostgreSQL password
- `DB_NAME` - Database name (default: `aiv_db`)

### Cache

- `REDIS_PASSWORD` - **CHANGE THIS** - Redis auth password

### Logging

- `LOG_LEVEL` - Set to `DEBUG`, `INFO`, `WARNING`, or `ERROR`

---

## Production Deployment Tips

### 1. Use External PostgreSQL (Recommended)

For better reliability, use a managed PostgreSQL service:

```yaml
# In docker-compose.yml, remove the 'postgres' service and update main_app, etc.:
environment:
  - DATABASE_URL=postgresql://user:password@your-rds-host:5432/aiv_db
```

### 2. Use External Redis (Recommended)

```yaml
environment:
  - REDIS_URL=redis://:password@your-redis-host:6379/0
```

### 3. Enable SSL/TLS

Update `nginx.conf` with your SSL certificate:

```nginx
listen 443 ssl;
ssl_certificate /etc/nginx/certs/server.crt;
ssl_certificate_key /etc/nginx/certs/server.key;
```

Add to docker-compose.yml:

```yaml
volumes:
  - ./certs:/etc/nginx/certs:ro
```

### 4. Scale Inference Workers

For higher throughput, run multiple inference workers:

```bash
docker-compose up -d --scale ai_inference_worker=3
```

### 5. GPU Acceleration on Linux

The `devices` section in `docker-compose.yml` for `camera_server` and `ai_inference_worker` is enabled by default:

```yaml
devices:
  - /dev/dri:/dev/dri # Intel iGPU
  - /dev/kfd:/dev/kfd # AMD GPU
group_add:
  - video
  - render
```

**Note:** Not supported on Docker Desktop (Windows/Mac). You may need to comment this block out if deploying without GPU support.

### 6. Backup Volumes

```bash
docker run --rm -v ai-vigilance_pg_data:/data \
  -v /backup:/backup \
  alpine tar czf /backup/postgres_backup.tar.gz /data

docker run --rm -v ai-vigilance_redis_data:/data \
  -v /backup:/backup \
  alpine tar czf /backup/redis_backup.tar.gz /data
```

### 7. Monitor Container Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f main_app

# Last 50 lines
docker-compose logs --tail=50 analytics_worker
```

### 8. Update Application

```bash
docker-compose pull
docker-compose build --no-cache
docker-compose up -d
```

---

## Troubleshooting

### Services not starting?

Check logs:

```bash
docker-compose logs --tail=100 main_app
```

Common issues:

- Port 80 already in use → Change `ports: - "8080:80"` in Nginx
- Out of memory → Increase Docker memory allocation
- Database password mismatch → Verify DB_PASSWORD in .env

### Can't connect to cameras?

1. Ensure camera RTSP URLs are accessible from container
2. Check camera server logs: `docker-compose logs camera_server`
3. Test with: `ffmpeg -rtsp_transport tcp -i rtsp://camera-ip:554/stream`

### GPU not being used?

1. Verify driver: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
2. Check permissions: Ensure user is in `video` and `render` groups
3. Linux only: `devices` section won't work on Docker Desktop

### High memory usage?

1. Reduce Redis max memory in docker-compose.yml
2. Scale down inference workers
3. Increase PostgreSQL `shared_buffers` if DB is slow

---

## Stopping Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v

# Remove everything including images
docker-compose down -v --rmi all
```

---

## Performance Tuning

### PostgreSQL

Adjust in `docker-compose.yml`:

```yaml
    environment:
      POSTGRES_INITDB_ARGS: "-c shared_buffers=512MB -c max_connections=300 -c work_mem=10MB"
```

### Redis

Increase memory in `docker-compose.yml`:

```yaml
    command: redis-server --requirepass password --maxmemory 4gb
```

### Application Workers

Increase Uvicorn workers in main_app in `docker-compose.yml`:

```yaml
    command: uvicorn app:app --workers 8 --host 0.0.0.0 --port 9000
```

---

**Last Updated:** May 2026
