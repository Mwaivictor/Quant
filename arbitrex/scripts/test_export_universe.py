"""Test: read latest MT5 symbols JSON and generate trading universe CSV."""
import os
from arbitrex.raw_layer.cli import main as _cli

def run():
    # Use the CLI export-universe command programmatically
    import sys
    sys.argv = ['cli', 'export-universe']
    _cli()

if __name__ == '__main__':
    run()
