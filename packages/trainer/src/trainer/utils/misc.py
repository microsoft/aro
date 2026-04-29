from pathlib import Path
from typing import Any

import yaml


def list_yaml_files(directory: str) -> list[str]:
    return [f.stem for f in Path(directory).iterdir() if f.suffix == ".yaml"]


def project_root() -> Path:
    current_path = Path(__file__)
    for parent in [current_path, *current_path.parents]:
        if (parent / "uv.lock").exists():
            return parent
    raise FileNotFoundError(
        "Could not find 'uv.lock' in the current directory or any parent directories."
    )


def load_yaml(cfg: str) -> dict[str, Any]:
    cfg_file = project_root() / "config" / f"{cfg}.yaml"
    with cfg_file.open() as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        assert isinstance(cfg, dict)
    return cfg
