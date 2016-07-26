import json
from functools import lru_cache
from time import time

import requests

from utils import bfs, toposort, randrange, median, b58_to_int


C = 6
IPFS_API = 'http://127.0.0.1:5001/api/v0'


def ipfs(cmd, args):
    url = IPFS_URL+cmd+'?'+'&'.join('%s=%s' % (k, v)
                                    for k, v in args)
    r = requests.get(url)
    if r.status_code != 200:
        raise IOError("error while querying '%s'" % url)
    return r.text

def publish_event(ev):
    return ipfs('/object/put', {}) #TODO

def set_head(h):
    ipfs('/name/publish', (('arg', h),))

def get_head(c):
    return ipfs('/name/resolve', (('arg', c),)).split('/')[-1]

def majority(it):
    hits = [0, 0]
    for s, x in it:
        hits[int(x)] += s
    if hits[0] > hits[1]:
        return False, hits[0]
    else:
        return True, hits[1]

Event = namedtuple('Event', 'd p t c')


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
        ev = Event(None, (), Time(), self.id)
        h = publish_event(ev)
        set_head(h)
        self.tbd.add(h)
        self.height[h] = self.round[h] = 0
        self.witnesses[0][ev.c] = h
        self.can_see[h] = {ev.c: h}

    def hg(self, h):
        return load(ipfs('/object/get', (('arg', h), ('encoding', 'json')))))

    def is_valid_event(self, h):
        ev = self.hg(h)
        return (ev.p == ()
                or (len(ev.p) == 2
                    and self.hg(ev.p[0]).c == ev.c
                    and self.hg(ev.p[1]).c != ev.c))

    def sync(self, r_id, payload):
        """Return new event hashs in topological order."""


        remote_head = get_head(r_id)

        new_evs = tuple(reversed(bfs((remote_head,),
                lambda u: (p for p in self.hg(u).p if p not in self.height))))

        for u in new_evs:
            assert is_valid_event(u)

            ipfs('/pin/add', {'arg': '/ipfs/'+u})

            self.tbd.add(u)
            p = self.hg(u).p
            if p == ():
                self.height[u] = 0
            else:
                self.height[u] = max(self.height[x] for x in p) + 1

        ev = Event(payload, (get_head(self.id), remote_head), time(), self.id)
        h = publish_event(ev)
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
            ev = self.hg(h)
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

                if len(self.strongly_sees(h, r)) > self.min_s:
                    self.round[h] = r + 1
                else:
                    self.round[h] = r
                self.can_see[h][ev.c] = h
                if self.round[h] > self.round[ev.p[0]]:
                    self.witnesses[self.round[h]][ev.c] = h

    def iter_undetermined(self, max_c, r_):
        for r in range(max_c, r_):
            if r not in self.consensus:
                for w in self.witnesses[r].values():
                    if w not in self.famous:
                        yield r, w

    def iter_voters(self, max_c, max_r):
        for r_ in range(max_c + 1, max_r + 1):
            for w in self.witnesses[r_].values():
                yield r_, w

    def strongly_see(self, u, r):
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
        for r_, y in iter_voters(max_c, max_r):
            s = {self.witnesses[r_-1][c] for c in self.strongly_see(y, r_ - 1)}
            for r, x in iter_undetermined(max_c, r_):
                if r_ - r == 1:
                    self.votes[y][x] = x in s
                else:
                    v, t = majority((self.stake[self.hg(w).c], self.votes[w][x]) for w in s)
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
                            self.votes[y][x] = bool(self.hg(y).s[0] // 128)

        new_c = {r for r in done
                 if all(w in self.famous for w in self.witnesses[r].values())}
        self.consensus |= new_c
        return new_c


    def _earliest_ancestor(self, a, x):
        c = self.hg(x).c
        while (c in self.can_see[a] and self.higher(self.can_see[a][c], x)
               and self.hg(a).p):
            a = self.hg(a).p[0]
        return a

    def find_order(self, new_c):
        for r in sorted(new_c):
            f_w = {w for w in self.witnesses[r].values() if self.famous[w]}
            white = reduce(lambda a, b: a ^ b58_to_int(b), f_w, 0)
            ts = {}
            final = []

            for x in bfs((w for w in f_w if w in self.tbd),
                         lambda u: (p for p in self.hg(u).p if p in self.tbd)):

                c = self.hg(x).c
                s = {w for w in f_w if c in self.can_see[w]
                     and self.higher(self.can_see[w][c], x)}
                if sum(self.stake[self.hg(w).c] for w in s) > self.tot_stake / 2:
                    ts = median([self.hg(self._earliest_ancestor(w, x)).t for w in s])
                    final.append((ts, white ^ b58_to_int(x), x)

            final.sort()
            for i, x in enumerate(final):
                self.tbd.remove(x[2])
                self.idx[x[2]] = i + len(self.transactions)
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
