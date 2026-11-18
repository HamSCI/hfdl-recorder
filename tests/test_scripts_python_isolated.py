"""Root-run scripts must start Python in isolated mode (-I).

install.sh and deploy.sh run as root from whatever directory the operator
is in.  `python3 -c ...` puts that directory first on sys.path, so a module
planted there (config.py, json.py) would run as root.  -I leaves out the
cwd, PYTHONPATH and the user site; site-packages and the editable install
still load.  A call run as the service user (`sudo -u ...`) is exempt."""
import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CALL = re.compile(r'python3?"?\s+(-[a-zA-Z]*\s+)*-(c|m)\b')


def _root_calls(path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if CALL.search(code) and "sudo -u" not in code:
            yield n, code


@pytest.mark.parametrize("name", ["install.sh", "deploy.sh"])
def test_root_python_calls_are_isolated(name):
    path = SCRIPTS / name
    if not path.exists():
        pytest.skip(f"no {name}")
    bad = [(n, c.strip()) for n, c in _root_calls(path)
           if not re.search(r'python3?"?\s+(-[a-zA-Z]*\s+)*-I\b', c)]
    assert not bad, f"{name}: python run as root without -I: {bad}"
