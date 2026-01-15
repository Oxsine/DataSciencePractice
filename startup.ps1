
if ($PSVersionTable.PSVersion.Major -lt 6 -or $IsWindows) {
}
else {
    Write-Host "This script works only on Windows!"
    pause
    exit 1
}

try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion"
}
catch {
    Write-Host "Python not found! Install Python and add to PATH."
    pause
    exit 1
}

$venvPath = ".\.venv_ds"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

Write-Host "Activating virtual environment..."
& "$venvPath\Scripts\activate.ps1"

$currentPython = (where.exe python)[0]
if ($currentPython -like "*$venvPath*") {
    Write-Host "Success! Virtual environment activated."
} else {
    Write-Host "Warning: Virtual environment might not be activated."
    Write-Host "Run manually: .\.venv_ds\Scripts\activate.ps1"
}

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

Write-Host "Done!"
pause