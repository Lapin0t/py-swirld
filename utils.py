from collections import deque

from pysodium import randombytes


def median(l):
    l.sort()
    return .5*(times[len(l)//2]+times[(len(l)+1)//2])


alphabet = {c: i for i, c in enumerate(
    '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ')}

def b58_to_int(s):
    n = 0
    for c in s:
        n = 58 * n + alphabet[c]
    return n


def toposort(nodes, parents):
    seen = {}
    def visit(u):
        if u in seen:
            if seen[u] == 0:
                raise ValueError('not a DAG')
        elif u in nodes:
            seen[u] = 0
            for v in parents(u):
                yield from visit(v)
            seen[u] = 1
            yield u
    for u in nodes:
        yield from visit(u)


def bfs(s, succ):
    s = tuple(s)
    seen = set(s)
    q = deque(s)
    while q:
        u = q.popleft()
        yield u
        for v in succ(u):
            if not v in seen:
                seen.add(v)
                q.append(v)


def dfs(s, succ):
    seen = set()
    q = [s]
    while q:
        u = q.pop()
        yield u
        seen.add(u)
        for v in succ(u):
            if v not in seen:
                q.append(v)


def randrange(n):
    a = (n.bit_length() + 7) // 8  # number of bytes to store n
    b = 8 * a - n.bit_length()     # number of shifts to have good bit number
    r = int.from_bytes(randombytes(a), byteorder='big') >> b
    while r >= n:
        r = int.from_bytes(randombytes(a), byteorder='big') >> b
    return r
