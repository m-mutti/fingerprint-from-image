"""Allow ``python -m fingerprint_from_image`` as an alias of ``fingerprint-match``."""

from .cli.match import main

raise SystemExit(main())
