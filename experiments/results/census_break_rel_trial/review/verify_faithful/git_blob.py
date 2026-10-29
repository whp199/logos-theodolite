#!/usr/bin/env python3
"""Read-only: fetch backend/census.py as committed at e94eea6 and at 372baec straight from loose git objects (zlib),
no git command run, nothing written under the repo. Writes the two blobs and a unified diff (against each other and
against the worktree file) into this folder."""
import difflib, hashlib, sys, zlib
from pathlib import Path
OBJ = Path('/home/eric/bin/ai/theodolite2/.git/objects')
HERE = Path(__file__).resolve().parent
WT = Path('/home/eric/bin/ai/theodolite2/.claude/worktrees/census-break-rel/backend/census.py')

def obj(sha):
    p = OBJ / sha[:2] / sha[2:]
    if not p.exists():
        raise SystemExit(f'object {sha} is not loose')
    raw = zlib.decompress(p.read_bytes())
    hdr, _, body = raw.partition(b'\0')
    assert hashlib.sha1(raw).hexdigest() == sha
    return hdr.split()[0].decode(), body

def tree_entries(body):
    i, out = 0, {}
    while i < len(body):
        sp = body.index(b' ', i); nul = body.index(b'\0', sp)
        name = body[sp + 1:nul].decode(); sha = body[nul + 1:nul + 21].hex()
        out[name] = sha; i = nul + 21
    return out

def path_blob(commit, path):
    t, body = obj(commit)
    assert t == 'commit'
    tree = body.split(b'\n')[0].split()[1].decode()
    for part in path.split('/'):
        t, b = obj(tree)
        assert t == 'tree'
        tree = tree_entries(b)[part]
    t, b = obj(tree)
    assert t == 'blob'
    return tree, b

res = {}
for c in ('e94eea672c8264c74c3538b08b482d3b03a18c22', '372baecfa60b7931096cafd1358f7df75bbc8934'):
    sha, b = path_blob(c, 'backend/census.py')
    (HERE / f'census_{c[:7]}.py').write_bytes(b)
    res[c[:7]] = b.decode()
    print(c[:7], 'blob', sha, len(b))
wt = WT.read_text()
print('worktree == 372baec:', wt == res['372baec'])
d = list(difflib.unified_diff(res['e94eea6'].splitlines(), wt.splitlines(), 'e94eea6', 'worktree', lineterm='', n=0))
(HERE / 'census_e94eea6_vs_worktree.diff').write_text('\n'.join(d) + '\n')
print('\n'.join(d))
