# Neural Aquarium Deployment Guide

## Local Development

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Neural DSL installed (`pip install -e .` from repo root)

### Quick Start

**Option 1: Automated (Recommended)**
```bash
cd neural/aquarium
./start-dev.sh        # Linux/Mac
start-dev.bat         # Windows
```

**Option 2: Manual**

Terminal 1 (Backend):
```bash
cd neural/aquarium/backend
python api.py
```

Terminal 2 (Frontend):
```bash
cd neural/aquarium
npm install
npm start
```

Access at: http://localhost:3000

## Production Deployment

### Frontend Build

1. **Build React App**
```bash
cd neural/aquarium
npm run build
```

2. **Output**: `build/` directory contains static files

3. **Serve Options**:
   - Nginx
   - Apache
   - AWS S3 + CloudFront
   - Netlify
   - Vercel

### Backend Deployment

#### Using Gunicorn (Recommended)

1. **Install Gunicorn**
```bash
pip install gunicorn
```

2. **Run**
```bash
cd neural/aquarium/backend
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

#### Using Docker

**Dockerfile** (backend):
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY neural/aquarium/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY neural/ ./neural/

WORKDIR /app/neural/aquarium/backend
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api:app"]
```

**Build & Run**:
```bash
docker build -t neural-aquarium-backend .
docker run -p 5000:5000 neural-aquarium-backend
```

#### Docker Compose (Full Stack)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped
```

**Dockerfile.frontend**:
```dockerfile
FROM node:16 as build

WORKDIR /app
COPY neural/aquarium/package*.json ./
RUN npm install
COPY neural/aquarium/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf**:
```nginx
server {
    listen 80;
    server_name localhost;
    
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://backend:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Cloud Deployment

#### AWS

**Frontend (S3 + CloudFront)**:
1. Build: `npm run build`
2. Upload `build/` to S3 bucket
3. Enable static website hosting
4. Configure CloudFront distribution

**Backend (EC2 or ECS)**:
1. Launch EC2 instance or ECS task
2. Install dependencies
3. Run with Gunicorn
4. Configure security groups for port 5000

#### Heroku

**Backend**:
```bash
cd neural/aquarium/backend
echo "web: gunicorn api:app" > Procfile
heroku create neural-aquarium-api
git push heroku main
```

**Frontend**:
```bash
cd neural/aquarium
heroku create neural-aquarium-frontend
heroku buildpacks:set mars/create-react-app
git push heroku main
```

#### Google Cloud Platform

**Frontend (Cloud Storage + CDN)**:
```bash
gsutil mb gs://neural-aquarium-frontend
gsutil -m cp -r build/* gs://neural-aquarium-frontend
gsutil web set -m index.html gs://neural-aquarium-frontend
```

**Backend (Cloud Run)**:
```bash
gcloud run deploy neural-aquarium-api \
  --source=neural/aquarium/backend \
  --port=5000 \
  --allow-unauthenticated
```

#### Azure

**Frontend (Static Web Apps)**:
```bash
az staticwebapp create \
  --name neural-aquarium \
  --resource-group neural-rg \
  --source neural/aquarium \
  --location centralus
```

**Backend (App Service)**:
```bash
az webapp up \
  --name neural-aquarium-api \
  --runtime PYTHON:3.9 \
  --sku B1
```

## Environment Configuration

### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:5000
```

Production:
```env
REACT_APP_API_URL=https://api.neural-aquarium.com
```

### Backend (environment variables)
```bash
FLASK_ENV=production
FLASK_APP=api.py
PORT=5000
```

## Reverse Proxy Setup

### Nginx

```nginx
upstream backend {
    server localhost:5000;
}

server {
    listen 80;
    server_name neural-aquarium.com;
    
    location / {
        root /var/www/neural-aquarium/build;
        try_files $uri /index.html;
    }
    
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Apache

```apache
<VirtualHost *:80>
    ServerName neural-aquarium.com
    
    DocumentRoot /var/www/neural-aquarium/build
    
    <Directory /var/www/neural-aquarium/build>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    ProxyPass /api http://localhost:5000/api
    ProxyPassReverse /api http://localhost:5000/api
</VirtualHost>
```

## SSL/HTTPS Setup

### Let's Encrypt (Certbot)

```bash
sudo certbot --nginx -d neural-aquarium.com
```

### Manual SSL

```nginx
server {
    listen 443 ssl http2;
    server_name neural-aquarium.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # ... rest of config
}
```

## Monitoring & Logging

### Backend Logging

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Frontend Error Tracking

Consider integrating:
- Sentry
- LogRocket
- Datadog

### Health Checks

**Backend**:
```bash
curl http://localhost:5000/health
```

**Frontend**:
```bash
curl http://localhost:3000
```

## Performance Tuning

### Backend
- Use Gunicorn with 4+ workers
- Enable gzip compression
- Implement caching (Redis)
- Database connection pooling

### Frontend
- Code splitting
- Lazy loading
- CDN for static assets
- Browser caching headers

## Security Checklist

- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] Input validation on all endpoints
- [ ] Rate limiting implemented
- [ ] Security headers configured
- [ ] Dependencies updated
- [ ] Environment variables secured
- [ ] API authentication (for production)

## Backup & Recovery

### Database (if added)
```bash
# Backup
pg_dump neural_aquarium > backup.sql

# Restore
psql neural_aquarium < backup.sql
```

### Configuration
- Version control all config files
- Document environment variables
- Keep backup of SSL certificates

## Troubleshooting

### Backend won't start
- Check Python version: `python --version`
- Verify dependencies: `pip list`
- Check port availability: `netstat -an | grep 5000`

### Frontend won't build
- Clear cache: `npm cache clean --force`
- Delete node_modules: `rm -rf node_modules && npm install`
- Check Node version: `node --version`

### API connection fails
- Verify CORS settings
- Check firewall rules
- Confirm backend is running
- Test with curl

## Scaling

### Horizontal Scaling
- Load balancer (Nginx, HAProxy)
- Multiple backend instances
- Session management (Redis)

### Vertical Scaling
- Increase worker count
- Optimize queries
- Add caching layer

## Maintenance

### Updates
```bash
# Frontend
npm update
npm audit fix

# Backend
pip install --upgrade -r requirements.txt
```

### Health Monitoring
- Set up uptime monitoring (UptimeRobot, Pingdom)
- Configure alerts for downtime
- Monitor error rates

## Cost Optimization

- Use CDN for frontend assets
- Implement caching strategies
- Auto-scaling for backend
- Use spot instances (AWS)
- Monitor resource usage
