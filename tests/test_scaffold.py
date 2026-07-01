"""Basic scaffold tests to verify project structure."""
import pytest
import sys
from pathlib import Path


def test_imports():
    """Test that all package namespaces can be imported."""
    import finops
    import finops.db
    import finops.modules
    import finops.daemon
    import finops.cli
    assert True


def test_cli_main_exists():
    """Test that the CLI main module exists."""
    from finops.cli import main
    assert hasattr(main, 'cli')


@pytest.mark.asyncio
async def test_async_support():
    """Test that async/await works with pytest-asyncio in auto mode."""
    async def dummy_coro():
        return True

    result = await dummy_coro()
    assert result is True


def test_pyproject_settings():
    """Verify pyproject.toml critical settings."""
    import tomllib

    with open(Path(__file__).parent.parent / 'pyproject.toml', 'rb') as f:
        config = tomllib.load(f)

    # Check required settings
    assert config['project']['requires-python'] == '>=3.11'
    assert config['tool']['pytest']['ini_options']['asyncio_mode'] == 'auto'
    assert config['project']['scripts'].get('finops') == 'finops.cli.main:cli'
