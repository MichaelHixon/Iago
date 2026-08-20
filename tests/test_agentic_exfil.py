"""The shared agentic-exfil scaffold (ISC-19). The surface modules delegate their oracle, loader,
suite, and report here — so the no-network anti-claim must be guarded HERE too, not only in the
per-surface egress tests (which now only scan their own thin module).
"""

import inspect

import iago.agentic_exfil as ax


def test_shared_scaffold_imports_no_network_machinery():
    # A future edit that pulled a networked lib into the shared scaffold would otherwise ship green:
    # the per-surface getsource() egress tests no longer transitively cover this module.
    src = inspect.getsource(ax)
    for banned in ("import socket", "from socket", "import urllib", "from urllib",
                   "import requests", "import http", "socket.socket", "urlopen(",
                   "import subprocess", "from subprocess"):
        assert banned not in src, f"shared exfil scaffold must not reference {banned!r}"
