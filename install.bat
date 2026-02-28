@echo off
echo 🚀 Starting installation for cuhackit-26...

:: Check for Git
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: git is not installed.
    exit /b 1
)

:: Clone the repository if the folder doesn't exist
if not exist ".git" (
    echo Cloning repository...
    git clone https://github.com/Quantiset/cuhackit-26.git .
)

:: Determine project type and install dependencies
if exist "package.json" (
    echo 📦 Node.js project detected. Installing dependencies...
    call npm install
) else if exist "requirements.txt" (
    echo 🐍 Python project detected. Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
)

:: Handle environment variables
if exist ".env.example" (
    if not exist ".env" (
        echo ⚙️ Creating .env file...
        copy .env.example .env
    )
)

echo ✅ Installation complete!
pause
