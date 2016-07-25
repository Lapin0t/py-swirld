from random import shuffle

from bokeh.plotting import output_file, figure, show
from bokeh.palettes import plasma
from bokeh.models import FixedTicker, ColumnDataSource, HoverTool
from pysodium import crypto_sign_keypair

from utils import bfs, randrange
from swirld import Node

def plot(node):
    nodes = tuple(bfs((node.head,), lambda u: node.hg[u].p))
    memb = {c: i for i, c in enumerate(node.network)}
    indice = {u: i for i, u in enumerate(node.transactions)}

    colors = plasma(len(node.transactions))


    output_file('hash_graph.html')
    p = figure()

    p.xgrid.grid_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.minor_tick_line_color = None

    source = ColumnDataSource(data={'xs': [memb[node.hg[u].c] for u in nodes],
              'ys': [node.height[u] for u in nodes],
              'cs': [colors[indice.get(u, 0)] for u in nodes],
              'as': [1 if node.famous.get(u) else 0.4 for u in nodes],
              'las': [1 if node.famous.get(u) else 0 for u in nodes]})
    tr_rend = p.circle('xs', 'ys', size=8, source=source, color='cs', fill_alpha='as', line_alpha='las')
    print(tr_rend.data_source.data)
    show(p)


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


