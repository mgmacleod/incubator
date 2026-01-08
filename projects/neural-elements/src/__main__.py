"""
Entry point for running Neural Elements as a module.

Usage:
    python -m src.web.app    # Run the web interface
    python -m src             # Show help
"""

import sys


def main():
    print("""
Neural Elements - A Periodic Table of Neural Networks
======================================================

Usage:
    python -m src.web.app    Start the web interface at http://localhost:5000

Examples:
    python examples/quick_start.py       Quick start example
    python examples/explore_elements.py  Comprehensive exploration

For more information, see README.md
    """)


if __name__ == '__main__':
    main()
