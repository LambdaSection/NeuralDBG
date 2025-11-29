# Manual GitHub Publishing Guide

## Prerequisites
You need to create 4 GitHub repositories either via:
- **GitHub CLI** (recommended): `winget install GitHub.cli` then `gh auth login`
- **GitHub Web Interface**: https://github.com/new

## Projects to Publish

1. **Neural-Aquarium** - Tauri desktop IDE
2. **NeuralPaper** - Web visualization platform  
3. **Neural-Research** - Paper annotations
4. **Lambda-sec-Models** - Production models

## Option 1: Using GitHub CLI (Automated)

### Install GitHub CLI
```powershell
winget install GitHub.cli
```

### Authenticate
```powershell
gh auth login
```

### Run Publishing Script
```powershell
.\scripts\github_publish_simple.ps1
```

## Option 2: Manual Setup (Web Interface)

### For Each Project:

#### 1. Create Repository on GitHub
- Go to https://github.com/new
- Owner: **Lemniscate-world**  
- Repository name: `Neural-Aquarium` (or other project name)
- Visibility: **Public**
- Don't initialize with README (we have code already)
- Click "Create repository"

#### 2. Push Code from Local
```powershell
# Navigate to project
cd C:\Users\Utilisateur\Documents\Neural-Aquarium

# Add remote (replace with your repo URL)
git remote add origin https://github.com/Lemniscate-world/Neural-Aquarium.git

# Push code  
git branch -M main
git push -u origin main
```

#### 3. Repeat for Other Projects
- **NeuralPaper**: https://github.com/Lemniscate-world/NeuralPaper
- **Neural-Research**: https://github.com/Lemniscate-world/Neural-Research
- **Lambda-sec-Models**: https://github.com/Lemniscate-world/Lambda-sec-Models

## Complete Commands (Copy-Paste)

### Neural-Aquarium
```powershell
cd C:\Users\Utilisateur\Documents\Neural-Aquarium
git remote add origin https://github.com/Lemniscate-world/Neural-Aquarium.git
git branch -M main
git push -u origin main
cd ..
```

### NeuralPaper
```powershell
cd C:\Users\Utilisateur\Documents\NeuralPaper
git remote add origin https://github.com/Lemniscate-world/NeuralPaper.git
git branch -M main
git push -u origin main
cd ..
```

### Neural-Research
```powershell
cd C:\Users\Utilisateur\Documents\Neural-Research
git remote add origin https://github.com/Lemniscate-world/Neural-Research.git
git branch -M main
git push -u origin main
cd ..
```

### Lambda-sec-Models
```powershell
cd C:\Users\Utilisateur\Documents\Lambda-sec-Models
git remote add origin https://github.com/Lemniscate-world/Lambda-sec-Models.git
git branch -M main
git push -u origin main
cd ..
```

## After Publishing

### Update EXTRACTED_PROJECTS.md
Replace `[TBD]` with actual GitHub URLs:
- Neural-Aquarium: https://github.com/Lemniscate-world/Neural-Aquarium
- NeuralPaper: https://github.com/Lemniscate-world/NeuralPaper
- Neural-Research: https://github.com/Lemniscate-world/Neural-Research
- Lambda-sec-Models: https://github.com/Lemniscate-world/Lambda-sec-Models

### Clean Up Old Directories
```powershell
cd C:\Users\Utilisateur\Documents\Neural

# Remove old directories
Remove-Item -Recurse -Force "Aquarium"
Remove-Item -Recurse -Force "neuralpaper"
Remove-Item -Recurse -Force "paper_annotation"
Remove-Item -Recurse -Force "Lambda-sec Models"

# Commit cleanup
git add .
git commit -m "Removed extracted projects - now in separate repositories"
git push
```

## Verification

Check that all repos are live:
- https://github.com/Lemniscate-world/Neural-Aquarium
- https://github.com/Lemniscate-world/NeuralPaper  
- https://github.com/Lemniscate-world/Neural-Research
- https://github.com/Lemniscate-world/Lambda-sec-Models
