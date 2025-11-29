# Quick Distribution Reference

## TL;DR: One Command Release

```bash
# Full automated release (recommended)
python scripts/automation/master_automation.py --release --version-type patch
```

This single command replaces all 20 manual steps and takes ~15 minutes vs 2-3 hours.

## Versioning Rules

- **Patch (0.0.X)**: Every 15 bugs fixed  
- **Minor (0.X.0)**: Each new feature
- **Major (X.0.0)**: Stable, bug-free release

## What Gets Automated

✅ Version bumping  
✅ CHANGELOG from closed issues  
✅ Blog posts (Dev.to, Medium, GitHub)  
✅ Social media (5 X posts, LinkedIn)  
✅ GitHub release  
✅ PyPI upload  
✅ Git tags  

## What's Still Manual

- GIF/Video creation
- Community posting (Reddit, Discord, ProductHunt)
- DM to influencers (9 people on your list)
- YouTube/Vimeo content

## Environment Setup

```bash
# 1. Copy env template
cp .env.example .env

# 2. Fill in your credentials (NEVER commit .env!)
# PYPI_TOKEN=...
# TWITTER_API_KEY=...

# 3. For GitHub Actions, add secrets:
# Settings → Secrets and variables → Actions
```

## Project Extraction

```bash
# Extract ecosystem projects to Documents folder
.\scripts\extract_projects.ps1

# This creates:
# C:\Users\Utilisateur\Documents\Neural-Aquarium\
# C:\Users\Utilisateur\Documents\NeuralPaper\
# C:\Users\Utilisateur\Documents\Neural-Research\
# C:\Users\Utilisateur\Documents\Lambda-sec-Models\
```

## Weekly Workflow

**Monday-Tuesday**: Code + fix bugs  
**Wednesday**: Generate release if ready, **REVIEW** blog  
**Thursday**: **POST** blog + social media  
**Friday**: Community engagement, DMs  
**Weekend**: YouTube content, benchmarks  

## Full Documentation

- **Complete Plan**: [DISTRIBUTION_PLAN.md](DISTRIBUTION_PLAN.md)
- **Versioning**: [CONTRIBUTING.md](CONTRIBUTING.md#versioning-strategy)
- **Automation**: [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)
- **Ecosystem**: [EXTRACTED_PROJECTS.md](EXTRACTED_PROJECTS.md)
