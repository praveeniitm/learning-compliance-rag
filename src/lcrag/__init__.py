"""Learning Compliance RAG: grounded Q&A over regulations, policies and course catalogs."""

import os
import sys

# macOS: the faiss-cpu and torch wheels each bundle their own OpenMP runtime; running both
# multi-threaded in one process segfaults. Pin OpenMP to one thread (torch models run on the
# Apple GPU via MPS by default, so this costs little). Linux is unaffected.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
