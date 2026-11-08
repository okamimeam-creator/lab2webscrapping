# Lab 02 – Web Scraping, Feature Engineering, and EDA

## Files
- `lab02_solution.py`: complete runnable solution.
- `requirements.txt`: required Python packages.
- `data/`: CSV files are created here when the script runs.
- `outputs/`: charts are created here when the script runs.

## Run in VS Code (Windows PowerShell)

Open this folder in VS Code, then open **Terminal → New Terminal**.

### 1. Create and activate the virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the same terminal, then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages
```powershell
python -m pip install -r requirements.txt --no-cache-dir
```

### 3. Install the spaCy English model
```powershell
python -m spacy download en_core_web_sm
```

### 4. Run the complete lab
```powershell
python lab02_solution.py
```

The script downloads the five URLs listed in the lab, extracts the required features,
calculates sentiment, creates EDA plots, and saves CSV/PNG outputs.

## Important
- An internet connection is required for scraping and installing packages.
- The original blog pages may be unavailable or may change their HTML. The script
  reports pages it cannot retrieve and continues with the remaining pages.
- If all pages fail, check internet access and the URLs before rerunning.
- Do not upload `.venv/` to GitHub. It is excluded by `.gitignore`.
