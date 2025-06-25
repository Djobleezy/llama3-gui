import subprocess

def test_ruff():
    subprocess.check_call(['ruff', 'check', '--quiet', '.'])
