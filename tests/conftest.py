import sys, pathlib
# Ensure the project root is on sys.path for imports like `homeagent.*`
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
