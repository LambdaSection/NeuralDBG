import os
import subprocess

venv_python = os.path.join(".venv", "Scripts", "python.exe")
subprocess.call([venv_python, "-m", "pip", "install", "-e", "."])
subprocess.call([venv_python, "-m", "pip", "install", "-r", "requirements-dev.txt"])
