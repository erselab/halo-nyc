"""HALO-specific Bayesian flux inversion built on the goe-inversion framework.

This package contains *only* the HALO-specific glue — regridding the NYC emission
inventories onto the Jacobian grid, deriving per-receptor backgrounds, and a
driver that composes everything. The generic inverse-problem machinery lives in
the separate ``goe-inversion`` project and is used here purely by import; no HALO
code belongs in that project.

Importing this package makes ``goe`` and ``adapters`` importable. If they are not
already on the path (e.g. via ``pip install -e /path/to/goe-inversion``), set the
environment variable ``GOE_INVERSION_PATH`` to the framework's location, or rely
on the default below.
"""

from __future__ import annotations

import os
import sys

# Default location of the goe-inversion checkout; override with GOE_INVERSION_PATH.
_DEFAULT_GOE_PATH = "/Volumes/Expansion/goe-inversion"


def _ensure_framework_importable() -> None:
    try:
        import goe  # noqa: F401
        import adapters  # noqa: F401
        return
    except ImportError:
        path = os.environ.get("GOE_INVERSION_PATH", _DEFAULT_GOE_PATH)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
        # re-raise a clear error if it still cannot be found
        try:
            import goe  # noqa: F401
            import adapters  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Could not import the goe-inversion framework. Either "
                "`pip install -e <goe-inversion>` or set GOE_INVERSION_PATH to its "
                f"directory (tried {path!r})."
            ) from exc


_ensure_framework_importable()
