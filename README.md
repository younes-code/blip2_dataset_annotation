cat >> README.md << 'EOF'

## Setup / Environment

```bash
# Preferred way - Conda
conda env create -f environment.yml
conda activate <your-env-name>

# Alternative - pip
pip install -r requirements.txt