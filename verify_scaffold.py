#!/usr/bin/env python3
"""Verify scaffold is set up correctly without requiring installation."""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """Verify all package namespaces can be imported."""
    try:
        import finops
        import finops.db
        import finops.modules
        import finops.daemon
        import finops.cli
        print("✓ All package imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def verify_cli():
    """Verify CLI entry point exists."""
    try:
        from finops.cli.main import cli
        print("✓ CLI entry point accessible")
        return True
    except ImportError as e:
        print(f"✗ CLI entry point failed: {e}")
        return False

def verify_pyproject():
    """Verify pyproject.toml has correct settings."""
    try:
        with open(project_root / 'pyproject.toml', 'r') as f:
            content = f.read()

        checks = [
            ('requires-python = ">=3.11"' in content, "Python version >= 3.11"),
            ('asyncio_mode = "auto"' in content, "pytest-asyncio mode: auto"),
            ('finops = "finops.cli.main:cli"' in content, "CLI script entry point"),
        ]

        all_passed = True
        for check, desc in checks:
            if check:
                print(f"✓ {desc}")
            else:
                print(f"✗ {desc}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"✗ pyproject verification failed: {e}")
        return False

def verify_files():
    """Verify all required files exist."""
    required_files = [
        'pyproject.toml',
        'docker-compose.yml',
        '.env.example',
        'finops/__init__.py',
        'finops/cli/__init__.py',
        'finops/cli/main.py',
        'finops/db/__init__.py',
        'finops/modules/__init__.py',
        'finops/daemon/__init__.py',
        'tests/__init__.py',
        'tests/cli/__init__.py',
        'tests/db/__init__.py',
        'tests/modules/__init__.py',
        'tests/daemon/__init__.py',
    ]

    all_exist = True
    for f in required_files:
        path = project_root / f
        if path.exists():
            print(f"✓ {f} exists")
        else:
            print(f"✗ {f} missing")
            all_exist = False

    return all_exist

if __name__ == '__main__':
    print("=== Scaffold Verification ===\n")

    print("File verification:")
    files_ok = verify_files()

    print("\nImport verification:")
    imports_ok = verify_imports()

    print("\nCLI verification:")
    cli_ok = verify_cli()

    print("\nProject configuration verification:")
    config_ok = verify_pyproject()

    print("\n" + "="*30)
    if all([files_ok, imports_ok, cli_ok, config_ok]):
        print("✓ All verifications passed")
        sys.exit(0)
    else:
        print("✗ Some verifications failed")
        sys.exit(1)
