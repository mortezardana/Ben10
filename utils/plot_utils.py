# utils/plot_utils.py

from pathlib import Path
from datetime import datetime


def create_plot_dir(base_dir="plots") -> Path:
    """
    Creates a timestamped directory inside the plots folder.

    Returns:
    - Path object to the newly created directory
    """
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = base_path / timestamp
    plot_path.mkdir(parents=True, exist_ok=True)
    return plot_path
