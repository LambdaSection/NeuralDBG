# Simple script to extract ecosystem projects
# Run from Neural directory

$baseDir = "C:\Users\Utilisateur\Documents"
$projects = @("Aquarium", "neuralpaper", "paper_annotation", "Lambda-sec Models")
$newNames = @("Neural-Aquarium", "NeuralPaper", "Neural-Research", "Lambda-sec-Models")

Write-Host "Extracting Neural Ecosystem Projects..." -ForegroundColor Cyan

for ($i = 0; $i -lt $projects.Count; $i++) {
    $source = Join-Path $PWD $projects[$i]
    $dest = Join-Path $baseDir $newNames[$i]
    
    if (-not (Test-Path $source)) {
        Write-Host "Skipping $($projects[$i]) - not found"  -ForegroundColor Yellow
        continue
    }
    
    if (Test-Path $dest) {
        Write-Host "Skipping $($newNames[$i]) - already exists" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "Extracting $($newNames[$i])..." -ForegroundColor Green
    
    # Copy files
    Copy-Item -Path $source -Destination $dest -Recurse -Force
    
    # Initialize git
    Push-Location $dest
    git init
    git add .
    git commit -m "Extracted from Neural repository"
    Pop-Location
    
    # Create notice
    Set-Content -Path (Join-Path $source "MOVED.md") -Value "# PROJECT MOVED`n`nNew location: $dest"
    
    Write-Host "Done: $($newNames[$i])" -ForegroundColor Green
}

Write-Host "`nExtraction complete!" -ForegroundColor Cyan
Write-Host "Projects extracted to: $baseDir" -ForegroundColor White
