"""Run the project CLI from a checkout without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from customer_policy_verifier.cli import main

if __name__ == "__main__":
    main()
