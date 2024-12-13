#!/bin/bash

# Create main project directory
cd ..  # Move back one directory since we're already in linguistic-analyzer
rm -rf linguistic-analyzer  # Remove the existing directory
mkdir linguistic-analyzer
cd linguistic-analyzer

# Create directory structure
mkdir -p linguistic_analyzer/analyzers
mkdir -p linguistic_analyzer/data
mkdir -p tests
mkdir -p logs
mkdir -p output

# Initialize git
git init

# Create .gitignore (same as before)
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
logs/

# Output
output/
data/

# Other
.DS_Store
EOL

# Create README.md (same as before)
cat > README.md << EOL
# Linguistic Analyzer

A real-time linguistic analysis suite for text processing.

## Installation

1. Create a virtual environment:
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
\`\`\`

2. Install requirements:
\`\`\`bash
pip3 install -r requirements.txt
\`\`\`

3. Install required models:
\`\`\`bash
python3 -m spacy download en_core_web_sm
\`\`\`

## Usage

[Add usage instructions here]
EOL

# Create virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# Install requirements (one at a time to handle errors better)
pip3 install --upgrade pip
pip3 install numpy
pip3 install pandas
pip3 install scipy
pip3 install matplotlib
pip3 install seaborn
pip3 install networkx
pip3 install nltk
pip3 install spacy
pip3 install torch
pip3 install transformers

# Create requirements.txt
pip3 freeze > requirements.txt

# Git commands
git add .
git commit -m "Initial commit"

# Set git config (add your details here)
echo "Please enter your Git name:"
read andrewshowie
echo "Please enter your Git email:"
read ahowie@umd.edu

git config --global user.name "$gitname"
git config --global user.email "$gitemail"

# Instructions for connecting to GitHub
echo "
Next steps:

1. Create a new repository on GitHub at https://github.com/new

2. Connect your local repository to GitHub:
   git remote add origin https://github.com/andrewshowie/Multidimensional-idiographic-SLD-software

3. Push your code to GitHub:
   git push -u origin main
"