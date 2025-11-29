#!/usr/bin/env pwsh
# Script to extract Neural ecosystem projects to separate directories
# Run from: C:\Users\Utilisateur\Documents\Neural

Write-Host "Neural Ecosystem Project Extraction" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Define base directories
$baseDir = "C:\Users\Utilisateur\Documents"
$sourceDir = "$baseDir\Neural"

# Check if we're in the right place
if (-not (Test-Path "$sourceDir\.git")) {
    Write-Host "ERROR: Not in Neural repository root!" -ForegroundColor Red
    Write-Host "Please run from: $sourceDir" -ForegroundColor Yellow
    exit 1
}

# Projects to extract
$projects = @(
    @{Name="Neural-Aquarium"; SourcePath="Aquarium"; Description="Tauri Desktop IDE"},
    @{Name="NeuralPaper"; SourcePath="neuralpaper"; Description="Web Visualization Platform"},  
    @{Name="Neural-Research"; SourcePath="paper_annotation"; Description="Paper Annotations"},
    @{Name="Lambda-sec-Models"; SourcePath="Lambda-sec Models"; Description="λ-S Production Models"}
)

Write-Host "`nProjects to extract:" -ForegroundColor Yellow
foreach ($project in $projects) {
    Write-Host "  ✓ $($project.Name) from $($project.SourcePath)" -ForegroundColor Green
}

Write-Host "`nThis will:" -ForegroundColor Yellow
Write-Host "  1. Create new directories in $baseDir" -ForegroundColor White
Write-Host "  2. Copy project files (preserving structure)" -ForegroundColor White
Write-Host "  3. Initialize new git repositories" -ForegroundColor White
Write-Host "  4. Add deprecation notices in old locations" -ForegroundColor White
Write-Host "  5. Update EXTRACTED_PROJECTS.md with local paths" -ForegroundColor White

$confirm = Read-Host "`nContinue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

foreach ($project in $projects) {
    Write-Host "`n--- Extracting $($project.Name) ---" -ForegroundColor Cyan
    
    $targetDir = "$baseDir\$($project.Name)"
    $sourcePath = "$sourceDir\$($project.SourcePath)"
    
    # Check if source exists
    if (-not (Test-Path $sourcePath)) {
        Write-Host "⚠ Source not found: $sourcePath (skipping)" -ForegroundColor Yellow
        continue
    }
    
    # Create target directory
    if (Test-Path $targetDir) {
        Write-Host "⚠ Target already exists: $targetDir (skipping)" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "Creating $targetDir..." -ForegroundColor White
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    
    # Copy files
    Write-Host "Copying files..." -ForegroundColor White
    Copy-Item -Path "$sourcePath\*" -Destination $targetDir -Recurse -Force
    
    # Initialize git
    Write-Host "Initializing git repository..." -ForegroundColor White
    Push-Location $targetDir
    git init
    git add .
    git commit -m "Initial commit: Extracted from Neural repository

Project: $($project.Name)
Description: $($project.Description)
Source: Neural/$($project.SourcePath)
Extraction Date: $(Get-Date -Format 'yyyy-MM-dd')

Part of λ-S (Lambda-Section) startup ecosystem."
    Pop-Location
    
    # Create deprecation notice in old location
    $notice = @"
# ⚠️ PROJECT MOVED

This project has been extracted to a separate repository.

**New Location**: ``C:\Users\Utilisateur\Documents\$($project.Name)``

**Description**: $($project.Description)

This directory will be removed in a future cleanup.
See ``EXTRACTED_PROJECTS.md`` for more information.
"@
    
    Set-Content -Path "$sourcePath\MOVED.md" -Value $notice
    
    Write-Host "✓ $($project.Name) extracted successfully!" -ForegroundColor Green
}

# Update EXTRACTED_PROJECTS.md
Write-Host "`n--- Updating EXTRACTED_PROJECTS.md ---" -ForegroundColor Cyan
$extractedProjectsFile = "$sourceDir\EXTRACTED_PROJECTS.md"

if (Test-Path $extractedProjectsFile) {
    $content = Get-Content $extractedProjectsFile -Raw
    
    # Update local paths
    $updates = @"

## Local Directory Structure (Updated: $(Get-Date -Format 'yyyy-MM-dd'))

C:\Users\Utilisateur\Documents\
├── Neural\                    # Main DSL repository
├── Neural-Aquarium\          # Desktop IDE (extracted)
├── NeuralPaper\              # Web visualization (extracted)
├── Neural-Research\          # Paper annotations (extracted)
└── Lambda-sec-Models\        # λ-S production models (extracted)

### Git Initialization Status
"@
    
    foreach ($project in $projects) {
        $targetDir = "$baseDir\$($project.Name)"
        if (Test-Path "$targetDir\.git") {
            $updates += "`n- [x] **$($project.Name)**: Git initialized at ``$targetDir``"
        } else {
            $updates += "`n- [ ] **$($project.Name)**: Not initialized"
        }
    }
    
    Add-Content -Path $extractedProjectsFile -Value $updates
    Write-Host "✓ EXTRACTED_PROJECTS.md updated!" -ForegroundColor Green
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Extraction Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Review extracted projects in $baseDir" -ForegroundColor White
Write-Host "2. Create GitHub repositories for each project" -ForegroundColor White
Write-Host "3. Add remote origins and push to GitHub" -ForegroundColor White
Write-Host "4. Update EXTRACTED_PROJECTS.md with GitHub URLs" -ForegroundColor White
Write-Host "5. Update main Neural README.md with links" -ForegroundColor White
Write-Host "6. Remove old directories from Neural repository" -ForegroundColor White

Write-Host "`nExample GitHub setup:" -ForegroundColor Yellow
Write-Host "  cd C:\Users\Utilisateur\Documents\Neural-Aquarium" -ForegroundColor Gray
Write-Host "  gh repo create Lemniscate-world/Neural-Aquarium --public" -ForegroundColor Gray
Write-Host "  git remote add origin https://github.com/Lemniscate-world/Neural-Aquarium.git" -ForegroundColor Gray
Write-Host "  git push -u origin main" -ForegroundColor Gray
