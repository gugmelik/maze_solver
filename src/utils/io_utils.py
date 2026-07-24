import json
import os
from collections import OrderedDict
from pathlib import Path
import tempfile

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname):
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """
    Write the content to the given json file.

    Args:
        content (Any JSON-friendly): content to write.
        fname (str): filename of the json file.
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_json_atomic(content, fname):
    """
    Atomically replace a JSON file with the provided content.

    The temporary file is created next to the destination so ``os.replace``
    stays on the same filesystem.
    """
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wt",
            encoding="utf-8",
            dir=fname.parent,
            prefix=f".{fname.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(content, handle, indent=4, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, fname)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
