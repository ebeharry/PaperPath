from __future__ import annotations

import subprocess
from pathlib import Path


def compile_success(tex_path: str) -> bool:
    out_dir = str(Path(tex_path).parent)
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", out_dir, tex_path],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except FileNotFoundError:
        raise RuntimeError(
            "pdflatex not found; install TeX Live or MacTeX to use compile_success"
        )
    except subprocess.TimeoutExpired:
        return False
