# Script to create GitHub repos and push extracted projects
# Requires GitHub CLI (gh) to be installed and authenticated

$projects = @(
    @{Name="Neural-Aquarium"; Description="Tauri-based desktop IDE for Neural DSL"},
    @{Name="NeuralPaper"; Description="Interactive web platform for visualizing neural network models"},
    @{Name="Neural-Research"; Description="Historical neural network paper analysis"},
    @{Name="Lambda-sec-Models"; Description="Production models for lambda-S startup"}
)

$baseDir = "C:\Users\Utilisateur\Documents"
$org = "Lemniscate-world"

Write-Host "Creating GitHub Repositories and Pushing Code" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if gh is installed
try {
    gh --version | Out-Null
} catch {
    Write-Host "ERROR: GitHub CLI (gh) is not installed!" -ForegroundColor Red
    Write-Host "Install from: https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Not authenticated with GitHub!" -ForegroundColor Red
    Write-Host "Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

Write-Host "GitHub CLI authenticated ✓" -ForegroundColor Green

foreach ($project in $projects) {
    Write-Host "`n--- Processing $($project.Name) ---" -ForegroundColor Cyan
    
    $projectDir = Join-Path $baseDir $project.Name
    
    if (-not (Test-Path $projectDir)) {
        Write-Host "⚠ Project directory not found: $projectDir" -ForegroundColor Yellow
        continue
    }
    
    Push-Location $projectDir
    
    # Create GitHub repo
    Write-Host "Creating GitHub repository..." -ForegroundColor White
    $repoFullName = "$org/$($project.Name)"
    
    # Check if repo already exists
    $repoExists = gh repo view $repoFullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Repository already exists: $repoFullName" -ForegroundColor Yellow
    } else {
        # Create new repo
        gh repo create $repoFullName --public --description $project.Description
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Repository created: https://github.com/$repoFullName" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to create repository" -ForegroundColor Red
            Pop-Location
            continue
        }
    }
    
    # Add remote if not exists
    $remoteUrl = "https://github.com/$repoFullName.git"
    $currentRemote = git remote get-url origin 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Adding remote origin..." -ForegroundColor White
        git remote add origin $remoteUrl
    } else {
        Write-Host "Remote origin already exists" -ForegroundColor Yellow
    }
    
    # Push to GitHub
    Write-Host "Pushing to GitHub..." -ForegroundColor White
    git branch -M main
    git push -u origin main --force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Pushed to GitHub: https://github.com/$repoFullName" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to push" -ForegroundColor Red
    }
    
    Pop-Location
}

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "GitHub Publishing Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan

Write-Host "`nRepositories created at:" -ForegroundColor Yellow
foreach ($project in $projects) {
    Write-Host "  https://github.com/$org/$($project.Name)" -ForegroundColor White
}
