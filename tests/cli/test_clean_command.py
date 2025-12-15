import os
from pathlib import Path

from click.testing import CliRunner

from neural.cli import cli


def test_clean_removes_generated_artifacts(tmp_path):
    runner = CliRunner()
    cwd = Path.cwd()
    try:
        # Work in a temp dir so we don't touch repo files
        os.chdir(tmp_path)
        # Create generated-like files
        Path("sample_tensorflow.py").write_text("print('tf')\n")
        Path("sample_pytorch.py").write_text("print('pt')\n")
        Path("architecture.png").write_text("fake")
        Path("shape_propagation.html").write_text("fake")
        # Create cache dir
        (tmp_path / ".neural_cache").mkdir()
        # Dry-run first
        result = runner.invoke(cli, ["clean"])  # no --yes
        assert result.exit_code == 0
        assert "would remove" in result.output.lower()
        # Apply deletions including caches
        result = runner.invoke(cli, ["clean", "--yes", "--all"])
        assert result.exit_code == 0
        assert not Path("sample_tensorflow.py").exists()
        assert not Path("sample_pytorch.py").exists()
        assert not Path("architecture.png").exists()
        assert not Path("shape_propagation.html").exists()
        assert not Path(".neural_cache").exists()
    finally:
        os.chdir(cwd)


def test_clean_no_items(tmp_path):
    runner = CliRunner()
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["clean"])  # dry-run
        assert result.exit_code == 0
        assert "no generated artifacts" in result.output.lower()
    finally:
        os.chdir(cwd)

