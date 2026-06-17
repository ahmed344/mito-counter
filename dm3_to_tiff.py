#!/usr/bin/env python3
from __future__ import annotations

from c3.dm3_to_tiff import main


def run() -> None:
    """Run the Calpaine_3 DM3 converter compatibility entry point.

    Args:
        None

    Returns:
        None: This function is executed for side effects only.
    """
    main()


if __name__ == "__main__":
    run()
