#!/bin/bash

echo "=== AI Study MVP Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create directory structure

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env
*.db
uploads/*
!uploads/.gitkeep
.DS_Store
*.log
.idea/
.vscode/
*.swp
*.swo
*~
EOF

# Create uploads/.gitkeep
touch uploads/.gitkeep

# Copy .env.example
echo "Creating .env file..."
if [ ! -f .env ]; then
    cp ../.env.example .env
    echo "⚠️  Please edit .env file and add your OPENAI_API_KEY"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r ./requirements.txt

# Check ffmpeg installation
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is installed"
else
    echo "⚠️  ffmpeg is not installed. Please install it:"
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
fi

# Create run script
echo "Creating run.sh script..."
cat > run.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python app.py
EOF
chmod +x run.sh

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY"
echo "2. Install ffmpeg if not already installed"
echo "3. Copy Python files (app.py, ml.py) to this directory"
echo "4. Copy HTML templates to templates/ directory"
echo "5. Run: ./run.sh"
echo ""
echo "For production deployment with Gunicorn:"
echo "   gunicorn -w 4 -b 0.0.0.0:5000 app:app"