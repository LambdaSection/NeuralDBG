# PowerShell script to run Neural API server with all services

Write-Host "🚀 Starting Neural API Server..." -ForegroundColor Green

# Check if Redis is running
try {
    $redisTest = redis-cli ping 2>&1
    if ($redisTest -notmatch "PONG") {
        throw "Redis not responding"
    }
    Write-Host "✅ Redis is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Redis is not running!" -ForegroundColor Red
    Write-Host "Please start Redis first:" -ForegroundColor Yellow
    Write-Host "  docker run -d -p 6379:6379 redis:7-alpine" -ForegroundColor Yellow
    exit 1
}

# Start Celery worker in background
Write-Host "🔧 Starting Celery worker..." -ForegroundColor Cyan
$celeryJob = Start-Job -ScriptBlock {
    celery -A neural.api.celery_app worker --loglevel=info --concurrency=4
}

# Wait for Celery to start
Start-Sleep -Seconds 3

# Start API server
Write-Host "🌐 Starting API server..." -ForegroundColor Cyan
$apiJob = Start-Job -ScriptBlock {
    uvicorn neural.api.main:app --host 0.0.0.0 --port 8000 --reload
}

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "✅ Neural API Server Started Successfully!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "API Server: http://localhost:8000" -ForegroundColor Yellow
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "To stop the server, press Ctrl+C" -ForegroundColor Yellow
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

# Handle shutdown
try {
    # Wait for jobs
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        if ((Get-Job -Id $celeryJob.Id).State -ne "Running" -and 
            (Get-Job -Id $apiJob.Id).State -ne "Running") {
            break
        }
    }
}
finally {
    Write-Host "Shutting down..." -ForegroundColor Yellow
    Stop-Job -Id $celeryJob.Id, $apiJob.Id -ErrorAction SilentlyContinue
    Remove-Job -Id $celeryJob.Id, $apiJob.Id -Force -ErrorAction SilentlyContinue
}
