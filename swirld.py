from collections import namedtuple, defaultdict, deque
from pickle import dumps, loads
from random import choice
from time import sleep

from pysodium import (crypto_sign_keypair, crypto_sign, crypto_sign_open,
                      crypto_sign_detached, crypto_sign_verify_detached,
                      crypto_generichash, randombytes)


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


"""
def bfs(s, succ, f, stop):
    seen = set()
    q = deque((x,))

    while q:
        u = q.popleft()
        yield u
        if stop():
            return
        seen.add(u)
        for v in succ(p):
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
"""

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
    elif hits[0] == hits[1]:
        return True, hits[1]
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
    def __init__(self, kp, network):
        self.pk, self.sk = kp
        self.network = network  # {pk -> Node.ask_sync} dict
        self.n = len(network) - 1
        self.min_s = 2 * self.n / 3  # min stake amount


        # {event-hash => event}: this is the hash graph
        self.hg = {}
        # event-hash: latest event from me
        self.head = None
        # {event-hash => round-num}: assigned round number of each event
        self.round = {}

        # {round-num}: rounds where famousness is fully decided
        self.consensus = set()
        # {event-hash => {event-hash => bool}}
        self.votes = {}
        # 
        self.recv_round = {}
        # {round-num => {member-pk => event-hash, trilean}}: 
        self.witnesses = defaultdict(dict)

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
        self.round[h] = 0
        self.witnesses[0][ev.c] = [h, Trilean.undetermined]
        self.seq[h] = 0

    def new_event(self, d, p):
        """Create a new event (and also return it's hash)."""

        assert len(p) == 2                   # 2 parents
        assert self.hg[p[0]].c == self.pk  # first exists and is self-parent
        assert self.hg[p[1]].c != self.pk  # second exists and not self-parent
        # TODO: fail if an ancestor of p[1] from creator self.pk is not an
        # ancestor of p[0]

        t = time()
        s = crypto_sign_detached(dumps((d, p, t, self.pk)), self.sk)
        ev = Event(d, p, t, self.pk, s)

        return crypto_generichash(dumps(ev)), ev

    def is_valid_event(self, h, ev):
        try:
            crypto_sign_verify_detached(dumps(ev[:-1]), ev.c)
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
        if ev.p == ():
            self.seq[h] = 0
        else:
            self.seq[h] = self.seq[ev.p[0]] + 1

    def sync(self, pk, payload):
        """Update hg and return new event ids in topological order."""

        msg = crypto_sign_open(self.network[pk](), pk)

        remote_head, remote_hg = loads(msg)
        new = toposort(remote_hg.keys() - self.hg.keys(),
                       lambda u: remote_hg[u].p)

        for h in new:
            ev = remote_hg[k]
            if self.is_valid_event(h, ev):
                self.add_event(h, ev)


        if self.is_valid(remote_head, remote_hg[remote_head]):
            h, ev = self.new_event(payload, (self.head, remote_head))
            # this really shouldn't fail, let's check it to be sure
            assert self.is_valid_event(h, ev)
            self.add_event(h, ev)
            self.head = h

        return new + [h]

    def ask_sync(self):
        """Respond to someone wanting to sync (only public method)."""

        # TODO: only send a diff? maybe with the help of self.seq
        # TODO: thread safe? (allow to run while mainloop is running)
        msg = dumps((self.head, self.hg))
        return crypto_sign(msg, self.sk) + msg

    def merge(it):
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
                if sum(1 for x in hits if x > self.min_s) > self.min_s:
                    self.round[h] = r + 1
                    self.witnesses[r + 1][ev.c] = [h, Trilean.undetermined]

                    # we need to start again the recurrence relation since h
                    # was promoted to the next round
                    self.can_see[h] = {ev.c: h}
                else:
                    self.round[h] = r

    def decide_fame(self):
        max_rounds = max(self.witnesses)
        for r in filter(lambda r: r not in self.consensus, range(max_rounds)):
            for x, fam in self.witnesses[r]:
                if fam != Trilean.undetermined:
                    continue
                for r_ in range(r+1, max_rounds+1):
                    for y in self.witnesses[r_].items():
                        # reconstruct seeable witnesses from previous round
                        can_see = self.merge(self.can_see[p] for p in ev.p
                                             if self.round[p] == r_-1)
                        can_see[self.hg[y].c] = y

                        if r_ - r == 1:
                            self.votes[y][x] = self.hg[x].c in can_see
                        else:
                            hits = defaultdict(int)
                            for k in can_see.values():
                                for c in self.can_see[k].keys():
                                    hits[c] += 1

                            s = {self.witnesses[r_ - 1][c] for c, n in hits
                                 if n > self.min_s}
                            t, v = majority(self.vote[w][x] for w in s)
                            if (r_ - r) % C != 0:
                                if t > self.min_s:
                                    self.witnesses[r][self.hg[x].c][1] = Trilean.true
                                    if all(w[1] != Trilean.undetermined for w in self.witnesses[r]):
                                        self.consensus.add(r)
                                    break
                                else:
                                    self.votes[y][x] = v
                            else:
                                if t > self.min_s:
                                    self.votes[y][x] = v
                                else:
                                    # 8-th bit is same as any other bit right?
                                    self.votes[y][x] = bool(self.hg[y].s[0] % 2)
                    # some ugly hack to break two loops at a time
                    else:
                        continue
                    break


    def main(self):
        """Main working loop."""

        while True:
            payload = (yield)
            # pick a random node to sync with but not me
            c = tuple(self.network.keys() - {self.pk})[randrange(self.n - 1)]
            new = self.sync(c, payload)

            self.divide_rounds(new)
            self.decide_fame()
            self.find_order()

def random_dag(n_nodes, n_syncs):
    top = [[0]] * n_nodes
    preds = {}
    for i in range(1, n_syncs):
        a, b = sample(range(n_nodes), 2)
        preds[i] = (top[a][-1], top[b][-1])
        top[a].append(i)
    return range(n_syncs), preds
