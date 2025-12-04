
python -m venv .venv_ds

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

if ($IsWindows) {
    # Windows
    if (Test-Path ".\.venv_ds\Scripts\Activate.ps1") {
        .\.venv_ds\Scripts\Activate.ps1
    }
}
else {
    # Linux/Mac
    if (Test-Path ".\.venv_ds\bin\activate") {
        .\.venv_ds\bin\activate
    }
}

# Проверка, что активация прошла успешно
if (-not (Test-Path (Join-Path $env:VIRTUAL_ENV "pyvenv.cfg"))) {
    Write-Host "Не удалось активировать виртуальное окружение"
    exit 1
}

# Установка зависимостей
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "Зависимости установлены успешно"
}
else {
    Write-Host "Файл requirements.txt не найден"
}