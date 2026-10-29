#!/usr/bin/env python3
"""Read-only: which files under backend/ differ between e94eea6 and 372baec (loose objects, no git command)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import git_blob_lib as G

def tree_of(commit, path):
    t, body = G.obj(commit)
    tree = body.split(b'\n')[0].split()[1].decode()
    for part in path.split('/'):
        t, b = G.obj(tree)
        tree = G.tree_entries(b)[part]
    return G.tree_entries(G.obj(tree)[1])

a = tree_of('e94eea672c8264c74c3538b08b482d3b03a18c22', 'backend')
b = tree_of('372baecfa60b7931096cafd1358f7df75bbc8934', 'backend')
print('backend files differing:', sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))
