# py-swirld

Just fooling around the _Swirld_ byzantine consensus algorithm by Leemon Baird
([whitepaper](http://www.swirlds.com/wp-content/uploads/2016/07/SWIRLDS-TR-2016-01.pdf)
available) in python.

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

## Work In Progress

- The interactive visualization is still rather crude.
- There is no strong fork detection when syncing.
- There is no stake (every node has stake 1).
- There is no real networking (the _communication_ is really just a method
  call). This should not be complicated to implement, but I will have to bring
  in threads, locks and stuff like that. I am actually thinking about embedding
  the hashgraph in [IPFS](http://ipfs.io/) objects as it fits perfectly (a
  hashgraph in IPFS nomenclature is a merkle DAG). This would enable to just
  drop any crypto and network operation as IPFS already takes care of it well.
