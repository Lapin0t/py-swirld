_Update: see [last paragraph](#some-updates) for some important status update._

# py-swirld

Just fooling around the _Swirlds_ byzantine consensus algorithm by Leemon Baird
([whitepaper](https://www.swirlds.com/downloads/SWIRLDS-TR-2016-01.pdf)
available) in python. _Swirlds_ is an algorithm constructing a strongly
consistent and partition tolerant, peer-to-peer append-only log.

It seems to work as intended to me but don't take it for granted!


## Dependencies

- python3 (obviously!)
- [pysodium](https://pypi.python.org/pypi/pysodium) for the crypto
- [bokeh](http://bokeh.pydata.org/en/latest/) for the analysis and interactive
  visualization

## Usage / High-level explainations

I don't think this is any useful to you if you don't plan to understand how the
algorithm works so you should read the whitepaper first. After that, the
implementation is _quite_ straightforward: the code is divided in the same
functions as presented in the paper:

- The main loop (which is a coroutine to enable step by step evaluation and
  avoid threads).
- `sync(<remote-node-id>, <payload-to-embed>)` which queries the remote node
  and updates local data.
- `divide_rounds` which sets round number and witnessness for the new
  transactions.
- `decide_fame` which does the voting stuff.
- `find_order` which update the final transactions list according to new
  election results.

Everything is packed into a `Node` class which is initialized with it's signing
keypair and with a dictionary mapping a node ID (it's public key) to some mean
to query data from him (the `ask_sync` method). Note that for simplicity, a
node is included in it's own mapping.

You can fiddle directly with that code or try out my nice interactive
visualizations to see how the network evolves in real time with:

```shell
bokeh serve --show viz.py --args <number of nodes>
```

This will start locally the specified number of nodes and by pressing the
_play_ button it will start choosing one at random every few miliseconds and do
a mainloop step. The color indicates the round number (it's just a random
color, the only thing is that transactions with the same round have the same
color).

## Algorithm details

Actually, I didn't implement the algorithm completely straitforward with full
graph traversals everywhere and big loops over all nodes. The main specificity
I introduced is a mapping I named `can_see`. It is updated along the round
number in `divide_rounds` and stores for each transaction a dictionnary that
maps to each node the id of the latest (highest) transaction from that node
this transaction can see (if there is one). It is easily updated by a
recurrence relation and enables to quickly seable and strongly seable
transactions.

With nn and nt respectively the number of nodes and the number of transactions,
this datastructure adds up O(nn\*nt) space and enables to compute the set of
witnesses a transaction can strongly see in O(nn^2).

## IPFS

A variant lives in the `ipfs` branch. This variant uses [IPFS](http://ipfs.io/)
as a backend to store the hashgraph. Indeed a swirlds _hashgraph_ is just the
same as an IPFS _merkle DAG_. This enables global deduplication of the
hashgraph (bandwith and computation efficient syncing between members). The
syncing process is just about getting the head of the remote member. As the
head of a member is stored in an IPNS record, this code is currently very slow,
but a lot of work is currently going on on the IPFS side to improve IPNS (_cf_
[IPRS](https://github.com/ipfs/go-iprs)).

## Work In Progress

- The interactive visualization is still rather crude.
- There is no strong fork detection when syncing.
- There is no real networking (the _communication_ is really just a method
  call). This should not be complicated to implement, but I will have to bring
  in threads, locks and stuff like that. I am actually thinking about embedding
  the hashgraph in [IPFS](http://ipfs.io/) objects as it fits perfectly. This
  would enable to just drop any crypto and network operation as IPFS already
  takes care of it well.

## Some updates

_AKA why swirlds isn't *that* much interesting_

Following [this issue](https://github.com/Lapin0t/py-swirld/issues/1), I want
to stress some things (also explaining why I stopped to be interested in
swirlds).

There are two flaws in this protocol which are somewhat related. The first one
is that the protocol cannot scale very well (ie have sublinear complexity in
the number of nodes for message handling) and the second one is that it doesn't
handle open-membership.

These issues may not be relevant if you want a distributed db in a medium-sized
closed organisation having some external centralized auth system. But this is
important if you want to make an internet-sized distributed db which has
absolutely no owner and no centralized registration service.

I'm gonna start with the second issue. To have a distributed db with
open-membership you *must* have protocol handling open-membership, you can't
make some construction with a second distributed db for the current stake
repartition with swirlds because there should then be a fixed set of "stake
validators". Some solutions proposed on swirlds.com are:

- an invite system where one gives a share of it's stake to people he's
  inviting, not trivial to do and the one starting the network has full power
  until he gives some away
- rely on some external mean (associating with a bitcoin wallet etc), that's
  aweful because this should be something replacing it
- use PoW for the stake, hurray, we just got back to a blockchain

They are all wacky because there is no mean to transform a closed membership
protocol into an open one.

The first issue gives a hint why it is probably not interesting to use any hack
or external mean to turn swirlds into an open-membership protocol: just try to
think of an efficient algorithm for the voting part (`divide_rounds`,
`decide_fame`, ...), you must in one way or another iterate through all members
(and probably also through some part of their respective transaction history)
so each transaction takes a time *at least* linear in the number of nodes. Sure
you can sync only with a few people, maybe O(log n) or even O(1) using a
carefully choosen De Bruijn graph but you will still need to maintain the full
hashgraph and iterate through witness transactions (and there is one for each
node), there's no escaping from the O(n) time and space lower bound and linear
complexity is bad(tm). Any decent distributed database that wants internet
scale should be having time and space bounds at most Omega(log n) for incoming
events processing.

So bottom line: don't star this repo, this algorithm is bad ;).

### Conclusion

Why do you actually need strong consensus? (tldr: you don't)

1. Probably you want to create a new cryptocurrency, but crypto-currencies are
   mostly shitty in the sense that the 2 groups interested in them are (1)
   speculators which are people how's job is to scam other people (yep, their
   job is mostly to buy things lower than what people think it's worth and
   resell it higher, having no interaction at all with the underlying physical
   object of their transactions), they only care about how much money will be
   in their pocket at the end. I don't think it has any sense to design an
   economic system where you can get money without creating anything, physical
   or intellectual) (2) right-libertarian who love capitalism: nope guys, the
   market isn't automagically stabilizing, it strives to make rich people
   richer and poor people poorer, which is quite the oposite of convergence.
   So at the end, i have nothing special against *crypto*-currencies per se,
   it's just that it has attracted quite a lot of attention from uninteresting
   people that have uninteresting and very conservative ideas.

2. Maybe you're writing this awesome service that is fully distributed (p2p)
   and you need some structure to handle shared mutable data. You should think
   twice about what real guarantees you need, because high chance are that you
   can overcome weak consensus. There are loads of distributed datastructures
   that take advantage of providing lower guarantees but still be on point for
   a particular use.

   - DHT will be great for a cache and if you're adventurous you can go read
     some papers about distributed skip-lists and skip-graph.
   - A lot of abstract datastructures (mostly sets and counters) have CRDT
     implementations which you can use. You can then plug an anti-entropy (or
     epidemic) protocol to synchronize the states in an eventually consistent
     fashion.
   - Networking is hard, if you need something like decentralized pub/sub, you
     should take a look at the Matrix protocol. It's highly unstable and I have
     some criticism on their protocol, but they're trying to solve this
     problem.

3. Oh so you do need strong consensus with open membership because you want to
   create a replacement protocol for naming things on the internet? Somewhat a
   mix between DNS and SSL-PKI? and distributed keybase? Sure, that's a good
   goal, but at first, you shouldn't rely on unique global identifiers that are
   not self-authentifying (so you shouldn't need strong consistency for that
   part). That said, strong consistency might be useful in this case. You
   should probably take a look at SCP (stellar consensus protocol), this one
   should be good.

### Post-Scriptum

Please don't ever use or mention blockchain protocols. If you encounter someone
who does, please repeat him the following (or just go away from him, chances
are he falls into one of the two categories of first conclusion bullet and he
will be much harder to deal with).

Blockchains are ledgers, not generic databases. Sure you can build a
generic database on top of a ledger, but i'm not sure we would like a world
where every database would be a dictionary mapping unique IDs to integers.

PoW blockchain's security claims rely on:

- The protocol's own inefficiency. The only slow part is block creation and
  making it fast would render the protocol trivially insecure.
- The fact that users rational. Maybe we are maybe not, the problem is that is
  just a fancy way of saying they assume users want to maximise the number of
  currency-tokens they have which is *not* a valid assumption: adversaries will
  surely not behave in a manner that maximises their tokens, they only want to
  crash your system. Instead you should target a byzantine-fault tolerant
  protocol.
