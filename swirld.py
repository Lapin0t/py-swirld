# -*- coding: utf-8 -*-

from collections import namedtuple, defaultdict
from pickle import dumps, loads
from random import choice
from time import time
from itertools import zip_longest
from functools import reduce

from pysodium import (crypto_sign_keypair, crypto_sign, crypto_sign_open,
                      crypto_sign_detached, crypto_sign_verify_detached,
                      crypto_generichash)

from utils import bfs, toposort, randrange


C = 6


def majority(it):
    hits = [0, 0]
    for s, x in it:
        hits[int(x)] += s
    if hits[0] > hits[1]:
        return False, hits[0]
    else:
        return True, hits[1]


Event = namedtuple('Event', 'd p t c s')
class Trilean:
    false = 0
    true = 1
    undetermined = 2


class Node:
    def __init__(self, kp, network, n_nodes, stake):
        self.pk, self.sk = kp
        self.network = network  # {pk -> Node.ask_sync} dict
        self.n = n_nodes
        self.stake = stake
        self.tot_stake = sum(stake.values())
        self.min_s = 2 * self.tot_stake / 3  # min stake amount


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
        self.idx = {}
        # {round-num}: rounds where famousness is fully decided
        self.consensus = set()
        # {event-hash => {event-hash => bool}}
        self.votes = defaultdict(dict)
        # {round-num => {member-pk => event-hash}}: 
        self.witnesses = defaultdict(dict)
        self.famous = {}

        # {event-hash => int}: 0 or 1 + max(height of parents) (only useful for
        # drawing, it may move to viz.py)
        self.height = {}
        # {event-hash => {member-pk => event-hash}}: stores for each event ev
        # and for each member m the latest event from m having same round
        # number as ev that ev can see
        self.can_see = {}

        # init first local event
        h, ev = self.new_event(None, ())
        self.add_event(h, ev)
        self.round[h] = 0
        self.witnesses[0][ev.c] = h
        self.can_see[h] = {ev.c: h}
        self.head = h

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
            self.height[h] = 0
        else:
            self.height[h] = max(self.height[p] for p in ev.p) + 1

    def sync(self, pk, payload):
        """Update hg and return new event ids in topological order."""

        info = crypto_sign(dumps({c: self.height[h]
                for c, h in self.can_see[self.head].items()}), self.sk)
        msg = crypto_sign_open(self.network[pk](self.pk, info), pk)

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

    def ask_sync(self, pk, info):
        """Respond to someone wanting to sync (only public method)."""

        # TODO: only send a diff? maybe with the help of self.height
        # TODO: thread safe? (allow to run while mainloop is running)

        cs = loads(crypto_sign_open(info, pk))

        subset = {h: self.hg[h] for h in bfs(
            (self.head,),
            lambda u: (p for p in self.hg[u].p
                       if self.hg[p].c not in cs or self.height[p] > cs[self.hg[p].c]))}
        msg = dumps((self.head, subset))
        return crypto_sign(msg, self.sk)

    def ancestors(self, c):
        while True:
            yield c
            if not self.hg[c].p:
                return
            c = self.hg[c].p[0]

    def maxi(self, a, b):
        if self.higher(a, b):
            return a
        else:
            return b

    def _higher(self, a, b):
        for x, y in zip_longest(self.ancestors(a), self.ancestors(b)):
            if x == b or y is None:
                return True
            elif y == a or x is None:
                return False

    def higher(self, a, b):
        return a is not None and (b is None or self.height[a] >= self.height[b])


    def divide_rounds(self, events):
        """Restore invariants for `can_see`, `witnesses` and `round`.

        :param events: topologicaly sorted sequence of new event to process.
        """

        for h in events:
            ev = self.hg[h]
            if ev.p == ():  # this is a root event
                self.round[h] = 0
                self.witnesses[0][ev.c] = h
                self.can_see[h] = {ev.c: h}
            else:
                r = max(self.round[p] for p in ev.p)

                # recurrence relation to update can_see
                p0, p1 = (self.can_see[p] for p in ev.p)
                self.can_see[h] = {c: self.maxi(p0.get(c), p1.get(c))
                                   for c in p0.keys() | p1.keys()}


                # count distinct paths to distinct nodes
                hits = defaultdict(int)
                for c, k in self.can_see[h].items():
                    if self.round[k] == r:
                        for c_, k_ in self.can_see[k].items():
                            if self.round[k_] == r:
                                hits[c_] += self.stake[c]
                # check if i can strongly see enough events
                if sum(1 for x in hits.values() if x > self.min_s) > self.min_s:
                    self.round[h] = r + 1
                else:
                    self.round[h] = r
                self.can_see[h][ev.c] = h
                if self.round[h] > self.round[ev.p[0]]:
                    self.witnesses[self.round[h]][ev.c] = h

    def decide_fame(self):
        max_r = max(self.witnesses)
        max_c = 0
        while max_c in self.consensus:
            max_c += 1

        # helpers to keep code clean
        def iter_undetermined(r_):
            for r in range(max_c, r_):
                if r not in self.consensus:
                    for w in self.witnesses[r].values():
                        if w not in self.famous:
                            yield r, w

        def iter_voters():
            for r_ in range(max_c + 1, max_r + 1):
                for w in self.witnesses[r_].values():
                    yield r_, w

        done = set()

        for r_, y in iter_voters():

            hits = defaultdict(int)
            for c, k in self.can_see[y].items():
                if self.round[k] == r_ - 1:
                    for c_, k_ in self.can_see[k].items():
                        if self.round[k_] == r_ - 1:
                            hits[c_] += self.stake[c]
            s = {self.witnesses[r_ - 1][c] for c, n in hits.items()
                 if n > self.min_s}

            for r, x in iter_undetermined(r_):
                if r_ - r == 1:
                    self.votes[y][x] = x in s
                else:
                    v, t = majority((self.stake[self.hg[w].c], self.votes[w][x]) for w in s)
                    if (r_ - r) % C != 0:
                        if t > self.min_s:
                            self.famous[x] = v
                            done.add(r)
                        else:
                            self.votes[y][x] = v
                    else:
                        if t > self.min_s:
                            self.votes[y][x] = v
                        else:
                            # the 1st bit is same as any other bit right?
                            self.votes[y][x] = bool(self.hg[y].s[0] // 128)

        new_c = {r for r in done
                 if all(w in self.famous for w in self.witnesses[r].values())}
        self.consensus |= new_c
        return new_c


    def find_order(self, new_c):
        to_int = lambda x: int.from_bytes(self.hg[x].s, byteorder='big')

        for r in sorted(new_c):
            f_w = {w for w in self.witnesses[r].values() if self.famous[w]}
            white = reduce(lambda a, b: a ^ to_int(b), f_w, 0)
            ts = {}
            seen = set()
            for x in bfs(filter(self.tbd.__contains__, f_w),
                         lambda u: (p for p in self.hg[u].p if p in self.tbd)):
                c = self.hg[x].c
                s = {w for w in f_w if c in self.can_see[w]
                                    and self.higher(self.can_see[w][c], x)}
                if sum(self.stake[self.hg[w].c] for w in s) > self.tot_stake / 2:
                    self.tbd.remove(x)
                    seen.add(x)
                    times = []
                    for w in s:
                        a = w
                        while (c in self.can_see[a]
                               and self.higher(self.can_see[a][c], x)
                               and self.hg[a].p):
                            a = self.hg[a].p[0]
                        times.append(self.hg[a].t)
                    times.sort()
                    ts[x] = .5*(times[len(times)//2]+times[(len(times)+1)//2])
            final = sorted(seen, key=lambda x: (ts[x], white ^ to_int(x)))
            for i, x in enumerate(final):
                self.idx[x] = i + len(self.transactions)
            self.transactions += final
        if self.consensus:
            print(self.consensus)



    def main(self):
        """Main working loop."""

        new = ()
        while True:
            payload = (yield new)

            # pick a random node to sync with but not me
            c = tuple(self.network.keys() - {self.pk})[randrange(self.n - 1)]
            new = self.sync(c, payload)
            self.divide_rounds(new)

            new_c = self.decide_fame()
            self.find_order(new_c)


def test(n_nodes, n_turns):
    kps = [crypto_sign_keypair() for _ in range(n_nodes)]
    network = {}
    stake = {kp[0]: 1 for kp in kps}
    nodes = [Node(kp, network, n_nodes, stake) for kp in kps]
    for n in nodes:
        network[n.pk] = n.ask_sync
    mains = [n.main() for n in nodes]
    for m in mains:
        next(m)
    for i in range(n_turns):
        r = randrange(n_nodes)
        print('working node: %i, event number: %i' % (r, i))
        next(mains[r])
    return nodes
