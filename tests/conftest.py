import os
import sys

# Make the top-level train_*/test_* scripts importable when running plain `pytest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
