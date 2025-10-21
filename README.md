========================================================

QUICK SETUP (Windows / Mac / Linux)

========================================================

1\. INSTALL UV (one time)

Windows PowerShell:

irm https://astral.sh/uv/install.ps1 | iex



Mac / Linux / Git Bash:

curl -LsSf https://astral.sh/uv/install.sh | sh



Then reopen your terminal and check:

uv --version



2\. CLONE THE REPOSITORY



git clone https://github.com/MateoLopez00/PR\_group7.git

cd PR\_group7



3\. CREATE THE ENVIRONMENT (from lock file)



uv sync --frozen



(This installs the exact Python version and packages like eveeryone else. Defined in pyproject.toml and uv.lock)



4\. RUN THE PROJECT



To launch Jupyter Lab:

uv run jupyter lab



To run a specific model (examples):

uv run python main.py --model svm --data\_root data

uv run python main.py --model mlp --data\_root data

uv run python main.py --model cnn --data\_root data

========================================================

DAILY COMMANDS SUMMARY

========================================================

Sync or update environment (use frozen to match exact versions):

uv sync --frozen



Run code with correct environment:

uv run python main.py --model mlp



Launch notebooks:

uv run jupyter lab

========================================================

TO ADD A NEW LIBRARY (only if needed)

========================================================

uv add <library-name>

uv lock

git add pyproject.toml uv.lock

git commit -m "Add <library-name> dependency"

git push



========================================================

TEAM ROLES

========================================================

SVM - Viola Meier

MLP - Marc Fuhrer \& Pascal André

CNN - Mateo Lopez \& Carlo Robbiani



