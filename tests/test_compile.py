import glob
import subprocess
import sys
import pytest

@pytest.mark.parametrize("path", glob.glob("app/*.py"))
def test_py_compile(path):
    subprocess.check_call([sys.executable, "-m", "py_compile", path])
