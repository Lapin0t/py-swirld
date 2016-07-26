import json
from functools import lru_cache
from time import time
from collections import namedtuple

import requests

from utils import bfs, toposort, randrange, median, b58_to_int


C = 6

IPFS_API = 'http://127.0.0.1:5001/api/v0'


def pub_event(ev):
    data = {'Data': json.dumps({'c': ev.c, 't': ev.t, 'd': ev.d}),
            'Links': [{'Name': '0', 'Hash': ev.p[0], 'Size': 0},
                      {'Name': '1', 'Hash': ev.p[1], 'Size': 0}]
                     if ev.p else []}
    r = requests.post(IPFS_API + '/object/put?inputenc=json',
            files={'data': ('foo', json.dumps(data), 'application/json')})
    if r.status_code != 200:
        raise IOError("couldn't publish event: %s" % r.text)
    return json.loads(r.text)["Hash"]


def set_head(h):
    r = requests.post(IPFS_API + '/name/publish?arg=' + h)
    if r.status_code != 200:
        raise IOError("couldn't set head: %s" % r.text)


def get_head(c):
    r = requests.get(IPFS_API + '/name/resolve?arg=' + c)
    if r.status_code != 200:
        raise IOError("couldn't get remote head: %s" % r.text)
    return r.text.split('/')[-1]

def get_event(h):
    r = requests.get(IPFS_API + '/object/get?encoding=json&arg=' + h)
    if r.status_code != 200:
        raise IOError("couldn't get event: %s" % r.text)
    obj = json.loads(r.text)
    data = json.loads(obj['Data'])
    return Event(p=tuple(l['Hash'] for l in obj['Links']), **data)


def parents(h):
    r = requests.get(IPFS_API + '/object/links?encoding=json&arg=' + h)
    if r.status_code != 200:
        raise IOError("couldn't get parents: %s" % r.text)
    obj = json.loads(r.text)
    return tuple(l['Hash'] for l in obj['Links'])


def pin_event(h):
    r = requests.get(IPFS_API + '/pin/add?arg=/ipfs/' + h)
    if r.status_code != 200:
        raise IOError("couldn't pin event: %s" % r.text)


def unpin_event(h):
    r = requests.get(IPFS_API + '/pin/rm?arg=/ipfs/' + h)
    if r.status_code != 200:
        raise IOError("couldn't unpin event: %s" % r.text)


def majority(it):
    hits = [0, 0]
    for s, x in it:
        hits[int(x)] += s
    if hits[0] > hits[1]:
        return False, hits[0]
    else:
        return True, hits[1]

Event = namedtuple('Event', 'p c t d')


def load(s):
    data = json.loads(s)
    data['p'] = tuple(data['p'])
    return Event(**data)


def dump(ev):
    return json.dumps(dict(d=ev.d, p=ev.p, t=ev.t, c=ev.c))


class Node:
    def __init__(self, multiaddrs, stake, id_):
        self.peers = tuple(m.split('/')[-1] for m in multiaddrs)
        self.stake = stake
        self.tot_stake = sum(stake.values())
        self.min_s = 2 * self.tot_stake / 3  # min stake amount
        self.id = id_

        self.round = {}
        self.tbd = set()
        self.transactions = []
        self.idx = {}
        self.consensus = set()
        self.votes = defaultdict(dict)
        self.witnesses = defaultdict(dict)
        self.famous = {}

        self.height = {}
        self.can_see = {}

        # init first local event
        h = pub_event(Event((), self.id, Time(), None))
        set_head(h)
        self.tbd.add(h)
        self.height[h] = 0
        self.round[h] = 0
        self.witnesses[0][self.id] = h
        self.can_see[h] = {self.id: h}

    def is_valid_event(self, h):
        ev = get_event(h)
        return (ev.p == ()
                or (len(ev.p) == 2
                    and get_event(ev.p[0]).c == ev.c
                    and get_event(ev.p[1]).c != ev.c))

    def sync(self, r_id, payload):
        """Return new event hashs in topological order."""


        remote_head = get_head(r_id)

        new_evs = tuple(reversed(bfs((remote_head,),
                lambda u: (p for p in parents(u) if p not in self.height))))

        for u in new_evs:
            assert is_valid_event(u)

            pin_event(u)

            self.tbd.add(u)
            p = parents(u)
            if p == ():
                self.height[u] = 0
            else:
                self.height[u] = max(self.height[x] for x in p) + 1

        h = pub_event(Event((get_head(self.id), remote_head),
                            self.id, time(), payload))
        set_head(h)

        return new + (h,)

    def maxi(self, a, b):
        if self.higher(a, b):
            return a
        else:
            return b

    def higher(self, a, b):
        return a is not None and (b is None or self.height[a] >= self.height[b])

    def divide_rounds(self, events):
        """Restore invariants for `can_see`, `witnesses` and `round`.

        :param events: topologicaly sorted sequence of new event to process.
        """

        for h in events:
            ev = get_event(h)
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
                self.can_see[h][ev.c] = h

                if len(self.strongly_seen(h, r)) > self.min_s:
                    self.round[h] = r + 1
                else:
                    self.round[h] = r
                if self.round[h] > self.round[ev.p[0]]:
                    self.witnesses[self.round[h]][ev.c] = h

    def _iter_undetermined(self, max_c, r_):
        for r in range(max_c, r_):
            if r not in self.consensus:
                for w in self.witnesses[r].values():
                    if w not in self.famous:
                        yield r, w

    def _iter_voters(self, max_c, max_r):
        for r_ in range(max_c + 1, max_r + 1):
            for w in self.witnesses[r_].values():
                yield r_, w

    def strongly_seen(self, u, r):
        for c, k in self.can_see[u].items():
            if self.round[k] == r:
                for c_, k_ in self.can_see[k].items():
                    if self.round[k_] == r:
                        hits[c_] += self.stake[c]
        return {c for c, n in hits.items() if n > self.min_s}

    def decide_fame(self):
        max_r = max(self.witnesses)
        max_c = 0
        while max_c in self.consensus:
            max_c += 1

        done = set()
        for r_, y in _iter_voters(max_c, max_r):
            s = {self.witnesses[r_-1][c] for c in self.strongly_seen(y, r_ - 1)}
            for r, x in _iter_undetermined(max_c, r_):
                if r_ - r == 1:
                    self.votes[y][x] = x in s
                else:
                    v, t = majority((self.stake[get_event(w).c], self.votes[w][x]) for w in s)
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
                            self.votes[y][x] = bool(b58_to_int(y) % 2)

        new_c = {r for r in done
                 if all(w in self.famous for w in self.witnesses[r].values())}
        self.consensus |= new_c
        return new_c


    def _earliest_ancestor(self, a, x):
        c = get_event(x).c
        while (c in self.can_see[a] and self.higher(self.can_see[a][c], x)
               and parents(a)):
            a = parents(a)[0]
        return a

    def find_order(self, new_c):
        for r in sorted(new_c):
            f_w = {w for w in self.witnesses[r].values() if self.famous[w]}
            white = reduce(lambda a, b: a ^ b58_to_int(b), f_w, 0)
            ts = {}
            final = []

            for x in bfs((w for w in f_w if w in self.tbd),
                         lambda u: (p for p in parents(u) if p in self.tbd)):

                c = get_event(x).c
                s = {w for w in f_w if c in self.can_see[w]
                     and self.higher(self.can_see[w][c], x)}
                if sum(self.stake[get_event(w).c] for w in s) > self.tot_stake / 2:
                    ts = median([get_event(self._earliest_ancestor(w, x)).t for w in s])
                    final.append((ts, white ^ b58_to_int(x), x))

            final.sort()
            for i, x in enumerate(final):
                u = x[2]
                self.tbd.remove(u)
                self.idx[u] = i + len(self.transactions)
                unpin
                del self.can_see[u]
            self.transactions += final

    def main(self):
        """Main working loop."""

        new = ()
        while True:
            payload = (yield new)

            # pick a random node to sync with but not me
            c = self.peers[randrange(len(self.peers))]
            new = self.sync(c, payload)
            self.divide_rounds(new)

            new_c = self.decide_fame()
            self.find_order(new_c)
