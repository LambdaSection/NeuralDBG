# Quick cleanup script - run AFTER publishing projects to GitHub

cd C:\Users\Utilisateur\Documents\Neural

Write-Host "Removing old project directories..." -ForegroundColor Yellow

Remove-Item -Recurse -Force "Aquarium" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "neuralpaper" -ErrorAction SilentlyContinue  
Remove-Item -Recurse -Force "paper_annotation" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "Lambda-sec Models" -ErrorAction SilentlyContinue

Write-Host "Old directories removed!" -ForegroundColor Green
Write-Host "Projects are now at:" -ForegroundColor Cyan
Write-Host "  https://github.com/Lemniscate-world/Neural-Aquarium"
Write-Host "  https://github.com/Lemniscate-world/NeuralPaper"
Write-Host "  https://github.com/Lemniscate-world/Neural-Research"
Write-Host "  https://github.com/Lemniscate-world/Lambda-sec-Models"

Write-Host "`nCommitting cleanup to main Neural repo..." -ForegroundColor Yellow
git add .
git commit -m "Removed extracted projects - now in separate repositories

Projects moved to:
- Neural-Aquarium: https://github.com/Lemniscate-world/Neural-Aquarium
- NeuralPaper: https://github.com/Lemniscate-world/NeuralPaper
- Neural-Research: https://github.com/Lemniscate-world/Neural-Research
- Lambda-sec-Models: https://github.com/Lemniscate-world/Lambda-sec-Models"

Write-Host "Done! Ready to push to GitHub." -ForegroundColor Green
