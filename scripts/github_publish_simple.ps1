# Simple GitHub publishing script
# Run this after installing GitHub CLI: winget install GitHub.cli

$projects = @("Neural-Aquarium", "NeuralPaper", "Neural-Research", "Lambda-sec-Models")
$baseDir = "C:\Users\Utilisateur\Documents"
$org = "Lemniscate-world"

Write-Host "Publishing to GitHub..." -ForegroundColor Cyan

# Check gh CLI
$ghPath = "gh"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    # Try common locations
    $commonPaths = @(
        "C:\Program Files\GitHub CLI\gh.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\gh.exe"
    )
    
    $found = $false
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $ghPath = $path
            $found = $true
            Write-Host "Found gh at: $ghPath" -ForegroundColor Green
            break
        }
    }
    
    if (-not $found) {
        Write-Host "ERROR: Install GitHub CLI: winget install GitHub.cli" -ForegroundColor Red
        Write-Host "If installed, please restart your terminal/VS Code." -ForegroundColor Yellow
        exit 1
    }
}

# Check auth
& $ghPath auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: & '$ghPath' auth login" -ForegroundColor Yellow
    exit 1
}

foreach ($proj in $projects) {
    Write-Host "`nProcessing $proj..." -ForegroundColor Green
    $dir = Join-Path $baseDir $proj
    
    if (-not (Test-Path $dir)) {
        Write-Host "Skipping $proj - not found"
        continue
    }
    
    Push-Location $dir
    
    # Create repo
    $repoName = "$org/$proj"
    & $ghPath repo view $repoName 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating repo..."
        & $ghPath repo create $repoName --public --confirm
    } else {
        Write-Host "Repo exists"
    }
    
    # Add remote
    git remote get-url origin 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        git remote add origin "https://github.com/$repoName.git"
    }
    
    # Push
    Write-Host "Pushing..."
    git branch -M main 
    git push -u origin main --force
    
    Write-Host "Done: https://github.com/$repoName" -ForegroundColor Green
    Pop-Location
}

Write-Host "`nAll projects published!" -ForegroundColor Cyan
