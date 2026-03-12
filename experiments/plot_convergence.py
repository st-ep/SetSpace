#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.darcy_1d.plot_convergence import branch_drift, main, synthetic_convergence


if __name__ == "__main__":
    main()
