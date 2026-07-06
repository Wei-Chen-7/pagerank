"""Enable ``python -m pagerank`` as an alias for the CLI."""

from .cli import main

raise SystemExit(main())
