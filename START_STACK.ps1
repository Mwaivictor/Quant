param([switch]$SkipRedis=$false,[int]$Port=8000)
Write-Host "Arbitrex Tick Stream Stack" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    exit 1
}
if (-not $env:VIRTUAL_ENV) {
    & ".venv\Scripts\Activate.ps1"
}
$env:TICK_QUEUE_BACKEND="redis"
$env:DISABLE_KAFKA="1"
$env:REDIS_URL="redis://localhost:6379/0"
Write-Host "Env configured" -ForegroundColor Green
if (-not $SkipRedis) {
    Write-Host "Checking Redis..." -ForegroundColor Yellow
    $exists=$false
    $ping=redis-cli ping 2>$null
    if ($ping -eq "PONG") {
        $exists=$true
    }
    if (-not $exists) {
        Write-Host "Starting Redis..." -ForegroundColor Yellow
        Start-Process redis-server -WindowStyle Minimized -NoNewWindow
        Start-Sleep -Seconds 2
        $ping=redis-cli ping 2>$null
        if ($ping -eq "PONG") {
            $exists=$true
        }
    }
    if (-not $exists) {
        Write-Host "Redis not available" -ForegroundColor Red
        exit 1
    }
    Write-Host "Redis OK" -ForegroundColor Green
}
Write-Host "Starting server..." -ForegroundColor Yellow
$python=".venv\Scripts\python.exe"
$proc=Start-Process $python -ArgumentList "-m arbitrex.scripts.run_streaming_stack" -PassThru -NoNewWindow
Start-Sleep -Seconds 5
Write-Host "Server started" -ForegroundColor Green
