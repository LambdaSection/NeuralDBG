# Install Neural DSL packages
$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "Installing core package in editable mode..."
& $venvPython -m pip install -e .

Write-Host "`nInstalling development dependencies..."
& $venvPython -m pip install -r requirements-dev.txt

Write-Host "`nSetup complete!"
