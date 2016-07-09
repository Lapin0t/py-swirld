from collections import deque, defaultdict
from random import sample
import matplotlib.pyplot as plt
from matplotlib import colors


def random_dag(n_nodes, n_syncs):
    top = [[i] for i in range(n_nodes)]
    preds = [()] * n_syncs
    for i in range(n_nodes + 1, n_syncs):
        a, b = sample(range(n_nodes), 2)
        preds[i] = (top[a][-1], top[b][-1])
        top[a].append(i)

    creator = list(range(n_nodes)) + [-1] * (n_syncs - n_nodes)
    seq = [0] * n_syncs
    for n, l in enumerate(top):
        for i, x in enumerate(l):
            seq[x] = i
            creator[x] = n

    return range(n_syncs), preds, seq, creator

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
    seen = set()
    q = deque((s,))
    while q:
        u = q.popleft()
        yield u
        seen.add(u)
        for v in succ(u):
            if not v in seen:
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


def strongly_see(succ, creator, seq, x, y):
    n_nodes = len(succ)
    n_members = len(set(creator))
    inv = [[] for _ in range(n_nodes)]
    for u, s in enumerate(succ):
        for p in s:
            inv[p].append(u)

    a = {}
    for k in bfs(x, succ.__getitem__):
        if creator[k] not in a:
            a[creator[k]] = seq[k]
        else:
            a[creator[k]] = max(seq[k], a[creator[k]])
    b = {}
    for k in bfs(y, inv.__getitem__):
        if creator[k] not in b:
            b[creator[k]] = seq[k]
        else:
            b[creator[k]] = min(seq[k], b[creator[k]])


    return sum(1 for i in a.keys() & b.keys() if a[i] >= b[i]) > 2 * n_members / 3


def rounds(succ, creator, seq, n_members):
    n_nodes = len(succ)
    rs = [0] * n_nodes
    for i in toposort(range(n_nodes), succ.__getitem__):
        if not succ[i]:
            rs[i] = 0
        else:
            r = max(rs[x] for x in succ[i])
            hits = set()
            for k in bfs(i, succ.__getitem__):
                if rs[k] == r and strongly_see(succ, creator, seq, i, k):
                    hits.add(creator[k])
                    if len(hits) > 2 * n_members / 3:
                        rs[i] = r + 1
                        break
            else:
                rs[i] = r
    return rs

def merge(seq, it):
    new = {}
    for d in it:
        for k, v in d.items():
            if k in new:
                new[k] = max(new[k], v, key=seq.__getitem__)
            else:
                new[k] = v
    return new


def rounds2(succ, creator, seq, n_members):
    n_nodes = len(succ)
    rs = [0] * n_nodes
    witnesses = defaultdict(dict)  # {round -> {creator -> seq}}
    can_see = dict()    # {event -> {creator -> seq}}
    for i in toposort(range(n_nodes), succ.__getitem__):
        if not succ[i]:
            rs[i] = 0
            witnesses[0][creator[i]] = i
            can_see[i] = {creator[i]:i}
        else:
            r = max(rs[x] for x in succ[i])

            # recurrence relation to update can_see
            can_see[i] = merge(seq, (can_see[p] for p in succ[i] if rs[p] == r))
            can_see[i][creator[i]] = i

            # count distinct paths to distinct nodes
            ns = [0] * n_members
            for k in can_see[i].values():
                for c in can_see[k].keys():
                    ns[c] += 1

            # check if i can strongly see enough events
            if sum(1 for x in ns if x > 2*n_members/3) > 2*n_members/3:
                rs[i] = r + 1
                witnesses[r + 1][creator[i]] = i
                can_see[i] = {creator[i]: i}
            else:
                rs[i] = r

    return rs

def plot(succ, seq, creator, n_members, r_fun=rounds, arrows=True,
         inline=False):
    n = len(succ)
    ax = plt.gca()
    rs = r_fun(succ, creator, seq, n_members)
    if inline:
        ys = seq
    else:
        ys = range(n)
    cs = list(colors.cnames)
    for i, (x, y) in enumerate(zip(creator, ys)):
        ax.add_artist(plt.Circle((x, y), .2, color=cs[rs[i]]))
    if arrows:
        for i, l in enumerate(succ):
            for s in l:
                plt.arrow(creator[i], ys[i], creator[s]-creator[i],
                          ys[s]-ys[i], head_width=0.1, head_length=0.1,
                          length_includes_head=True, color='k')
    plt.xlim(0, n_members)
    plt.ylim(0, n)
    plt.show()

