from collections import deque

from pysodium import randombytes

def from_bytes(b, byteorder='big'):
    if byteorder == 'big':
        a = [ord(c) for c in b[::-1]]
    else:
        a = [ord(c) for c in b]
    return sum(x**i for i, x in enumerate(a))

def toposort(nodes, parents):
    seen = {}
    def visit(u):
        if u in seen:
            if seen[u] == 0:
                raise ValueError('not a DAG')
        elif u in nodes:
            seen[u] = 0
            for v in parents(u):
                for w in visit(v):
                    yield w
            seen[u] = 1
            yield u
    for u in nodes:
        for v in visit(u):
            yield v


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
    r = from_bytes(randombytes(a), byteorder='big') >> b
    while r >= n:
        r = int.from_bytes(randombytes(a), byteorder='big') >> b
    return r
