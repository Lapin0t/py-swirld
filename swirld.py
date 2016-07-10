from collections import namedtuple, defaultdict, deque
from pickle import dumps, loads
from random import choice
from time import time
from math import atan2, sin, cos

from pysodium import (crypto_sign_keypair, crypto_sign, crypto_sign_open,
                      crypto_sign_detached, crypto_sign_verify_detached,
                      crypto_generichash, randombytes)
import matplotlib.pyplot as plt
from matplotlib import colors


C = 6


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


def randrange(n):
    a = (n.bit_length() + 7) // 8  # number of bytes to store n
    b = 8 * a - n.bit_length()     # number of shifts to have good bit number
    r = int.from_bytes(randombytes(a), byteorder='big') >> b
    while r >= n:
        r = int.from_bytes(randombytes(a), byteorder='big') >> b
    return r

def majority(it):
    hits = [0, 0]
    for x in it:
        hits[int(x)] += 1
    if hits[0] > hits[1]:
        return False, hits[0]
    else:
        return True, hits[1]



Event = namedtuple('Event', 'd p t c s')
class Trilean:
    false = 0
    true = 1
    undetermined = 2

class defaultlist(list):
    def __init__(self, f):
        self._f = f
        super().__init__()

    def __getitem__(self, idx):
        while len(self) <= idx:
            self.append(self._f())
        return super().__getitem__(idx)

    def __setitem__(self, idx, val):
        while len(self) <= idx:
            self.append(self._f())
        return super().__setitem__(idx, val)


class Node:
    def __init__(self, kp, network, n):
        self.pk, self.sk = kp
        self.network = network  # {pk -> Node.ask_sync} dict
        self.n = n
        self.min_s = 2 * self.n / 3  # min stake amount


        # {event-hash => event}: this is the hash graph
        self.hg = {}
        # event-hash: latest event from me
        self.head = None
        # {event-hash => round-num}: assigned round number of each event
        self.round = {}
        # {event-hash}: events for which final order remains to be determined
        self.tbd = set()
        # [event-hash]: final order of the transactions
        self.transactions = []
        # {round-num}: rounds where famousness is fully decided
        self.consensus = set()
        # {event-hash => {event-hash => bool}}
        self.votes = defaultdict(dict)
        # {event-hash => recv-round-num}: assigned 'received round' number
        self.recv_round = {}
        # {round-num => {member-pk => event-hash}}: 
        self.witnesses = defaultdict(dict)
        self.famous = {}

        # {round-num => {}}
        self.election = defaultdict(dict)

        # {event-hash => int}: stores the number of self-ancestors for each event
        self.seq = {}
        # {event-hash => {member-pk => event-hash}}: stores for each event ev
        # and for each member m the latest event from m having same round
        # number as ev that ev can see
        self.can_see = {}

        # init first local event
        h, ev = self.new_event(None, ())
        self.hg[h] = ev
        self.head = h
        self.round[h] = 0
        self.witnesses[0][ev.c] = h
        self.famous[h] = Trilean.undetermined
        self.seq[h] = 0
        self.can_see[h] = {ev.c: h}

    def new_event(self, d, p):
        """Create a new event (and also return it's hash)."""

        assert p == () or len(p) == 2                   # 2 parents
        assert p == () or self.hg[p[0]].c == self.pk  # first exists and is self-parent
        assert p == () or self.hg[p[1]].c != self.pk  # second exists and not self-parent
        # TODO: fail if an ancestor of p[1] from creator self.pk is not an
        # ancestor of p[0]

        t = time()
        s = crypto_sign_detached(dumps((d, p, t, self.pk)), self.sk)
        ev = Event(d, p, t, self.pk, s)

        return crypto_generichash(dumps(ev)), ev

    def is_valid_event(self, h, ev):
        try:
            crypto_sign_verify_detached(ev.s, dumps(ev[:-1]), ev.c)
        except ValueError:
            return False

        return (crypto_generichash(dumps(ev)) == h
                and (ev.p == ()
                     or (len(ev.p) == 2
                         and ev.p[0] in self.hg and ev.p[1] in self.hg
                         and self.hg[ev.p[0]].c == ev.c
                         and self.hg[ev.p[1]].c != ev.c)))

                         # TODO: check if there is a fork (rly need reverse edges?)
                         #and all(self.hg[x].c != ev.c
                         #        for x in self.preds[ev.p[0]]))))

    def add_event(self, h, ev):
        self.hg[h] = ev
        self.tbd.add(h)
        if ev.p == ():
            self.seq[h] = 0
        else:
            self.seq[h] = self.seq[ev.p[0]] + 1

    def sync(self, pk, payload):
        """Update hg and return new event ids in topological order."""

        msg = crypto_sign_open(self.network[pk](), pk)

        remote_head, remote_hg = loads(msg)
        new = tuple(toposort(remote_hg.keys() - self.hg.keys(),
                       lambda u: remote_hg[u].p))

        for h in new:
            ev = remote_hg[h]
            if self.is_valid_event(h, ev):
                self.add_event(h, ev)


        if self.is_valid_event(remote_head, remote_hg[remote_head]):
            h, ev = self.new_event(payload, (self.head, remote_head))
            # this really shouldn't fail, let's check it to be sure
            assert self.is_valid_event(h, ev)
            self.add_event(h, ev)
            self.head = h

        return new + (h,)

    def ask_sync(self):
        """Respond to someone wanting to sync (only public method)."""

        # TODO: only send a diff? maybe with the help of self.seq
        # TODO: thread safe? (allow to run while mainloop is running)
        msg = dumps((self.head, self.hg))
        return crypto_sign(msg, self.sk)

    def merge(self, it):
        """Merge to dicts by taking the highest event in case of conflict."""

        new = {}
        for d in it:
            for k, v in d.items():
                if k in new:
                    new[k] = max(new[k], v, key=self.seq.__getitem__)
                else:
                    new[k] = v
        return new

    def divide_rounds(self, events):
        """Restore invariants for `can_see`, `witnesses` and `round`.

        :param events: topologicaly sorted sequence of new event to process.
        """

        for h in events:
            ev = self.hg[h]
            if ev.p == ():  # this is a root event
                self.round[h] = 0
                self.witnesses[0][ev.c] = h
                self.famous[h] = Trilean.undetermined
                self.can_see[h] = {ev.c: h}
            else:
                r = max(self.round[p] for p in ev.p)

                # recurrence relation to update can_see
                self.can_see[h] = self.merge(self.can_see[p] for p in ev.p
                                             if self.round[p] == r)

                self.can_see[h][ev.c] = h

                # count distinct paths to distinct nodes
                hits = defaultdict(int)
                for k in self.can_see[h].values():
                    for c in self.can_see[k].keys():
                        hits[c] += 1

                # check if i can strongly see enough events
                if sum(1 for x in hits.values() if x > self.min_s) > self.min_s:
                    self.round[h] = r + 1
                    self.witnesses[r + 1][ev.c] = h
                    self.famous[h] = Trilean.undetermined

                    # we need to start again the recurrence relation since h
                    # was promoted to the next round
                    self.can_see[h] = {ev.c: h}
                else:
                    self.round[h] = r

    def decide_fame(self):
        max_round = max(self.witnesses)
        print('max round: ', max_round)

        # helpers to keep code clean
        def iter_undetermined():
            for r in filter(lambda r: r not in self.consensus,
                            range(max_round)):
                for w in self.witnesses[r].values():
                    if self.famous[w] == Trilean.undetermined:
                        yield r, w

        def iter_voters(r):
            for r_ in range(r + 1, max_round + 1):
                for w in self.witnesses[r_].values():
                    yield r_, w

        for r, x in iter_undetermined():
            for r_, y in iter_voters(r):
                # reconstruct seeable witnesses from previous round
                can_see = self.merge(self.can_see[p] for p in self.hg[y].p
                                     if self.round[p] == r_-1)

                if r_ - r == 1:
                    maxi = None
                    for u in bfs(y, lambda u: sorted(self.hg[u].p, key=self.seq.__getitem__, reverse=True)):
                        if self.hg[u].c == self.hg[x]:
                            maxi = u
                            break
                    self.votes[y][x] = maxi is not None and self.seq[maxi] >= self.seq[x]
                else:
                    hits = defaultdict(int)
                    for k in can_see.values():
                        for c in self.can_see[k].keys():
                            if c in self.witnesses[r_-1]:
                                hits[c] += 1

                    s = {self.witnesses[r_ - 1][c] for c, n in hits.items()
                         if n > self.min_s}
                    v, t = majority(self.votes[w][x] for w in s)
                    if (r_ - r) % C != 0:
                        if t > self.min_s:
                            if v:
                               self.famous[x] = Trilean.true
                            else:
                               self.famous[x] = Trilean.false
                            if all(self.famous[w] != Trilean.undetermined for w in self.witnesses[r].values()):
                                self.consensus.add(r)
                            break
                        else:
                            self.votes[y][x] = v
                    else:
                        if t > self.min_s:
                            self.votes[y][x] = v
                        else:
                            # the 8-th bit is same as any other bit right?
                            self.votes[y][x] = bool(self.hg[y].s[0] % 2)
            r = 0
            while (r not in self.consensus and len(self.witnesses[r]) > 0
                   and all(self.famous[w] != Trilean.undetermined for w in self.witnesses[r].values())):
                self.consensus.add(r)
                r += 1
            print('consensus: ', self.consensus)

    def main(self):
        """Main working loop."""

        while True:
            payload = (yield)
            # pick a random node to sync with but not me
            print('picking a node')
            c = tuple(self.network.keys() - {self.pk})[randrange(self.n - 1)]
            print('syncing...')
            new = self.sync(c, payload)
            assert all(p in self.seq for n in new for p in self.hg[n].p)

            print('dividing rounds...')
            self.divide_rounds(new)
            print('decinding fame...')
            self.decide_fame()
            #self.find_order()

    def plot(self, arrows=True, color='rounds'):
        nodes = {u:i for i, u in enumerate(bfs(self.head, lambda u: self.hg[u].p))}
        ax = plt.gca()
        cs = list(colors.cnames)
        cr = {c: i for i, c in enumerate(self.network)}
        if color == 'rounds':
            col = lambda u: 'red' if self.hg[u].c in self.witnesses[self.round[u]] and self.witnesses[self.round[u]][self.hg[u].c] == u else cs[self.round[u]]
        elif color == 'witness':
            def col(u):
                if self.hg[u].c in self.witnesses[self.round[u]]:
                    if self.witnesses[self.round[u]][self.hg[u].c] == u:
                        if self.famous[u] == Trilean.false:
                            return 'red'
                        elif self.famous[u] == Trilean.undetermined:
                            return 'orange'
                        else:
                            return 'green'
                return 'black'

        xs = lambda u: cr[self.hg[u].c]
        ys = self.seq.__getitem__
        if arrows:
            for u in nodes:
                for p in self.hg[u].p:
                    vect = (xs(p)-xs(u), ys(p) - ys(u))
                    alpha = atan2(vect[1], vect[0])
                    r = (vect[0]**2 + vect[1]**2)**.5 - .2
                    vect = (r*cos(alpha), r*sin(alpha))
                    plt.arrow(xs(u), ys(u), vect[0],
                              vect[1], head_width=0.1, head_length=0.2,
                              length_includes_head=True, color='gray')
        for u in nodes:
            ax.add_artist(plt.Circle((xs(u), ys(u)), .2 , color=col(u)))
        plt.xlim(-.5, self.n+.5)
        plt.ylim(-.5, max(self.seq.values())+.5)
        plt.show()


def test(n_nodes, n_turns):
    kps = [crypto_sign_keypair() for _ in range(n_nodes)]
    network = {}
    nodes = [Node(kp, network, n_nodes) for kp in kps]
    for n in nodes:
        network[n.pk] = n.ask_sync
    mains = [n.main() for n in nodes]
    for m in mains:
        next(m)
    for _ in range(n_turns):
        r = randrange(n_nodes)
        print('working node: ', r)
        next(mains[r])
    return nodes


