from random import shuffle
import sys

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.plotting import figure
from bokeh.palettes import plasma
from bokeh.models import (
        FixedTicker, Button, ColumnDataSource, PanTool, Scroll,
        RadioButtonGroup, RadioGroup, Arrow, NormalHead)
from pysodium import crypto_sign_keypair

from utils import bfs, randrange
from swirld import Node


R_COLORS = plasma(256)
shuffle(R_COLORS)
def round_color(r):
    return R_COLORS[r % 256]


class App:
    def __init__(self, n_nodes):
        kps = [crypto_sign_keypair() for _ in range(n_nodes)]

        network = {}
        self.nodes = [Node(kp, network, n_nodes) for kp in kps]
        for n in self.nodes:
            network[n.pk] = n.ask_sync
        self.ids = {kp[0]: i for i, kp in enumerate(kps)}

        self.main_its = [n.main() for n in self.nodes]
        for m in self.main_its:
            next(m)

        def toggle():
            if play.label == '► Play':
                play.label = '❚❚ Pause'
                curdoc().add_periodic_callback(self.animate, 50)
            else:
                play.label = '► Play'
                curdoc().remove_periodic_callback(self.animate)

        play = Button(label='► Play', width=60)
        play.on_click(toggle)

        def sel_node(new):
            self.active = new
            node = self.nodes[new]
            self.tr_src.data, self.links_src.data = self.extract_data(
                    self.nodes[new],
                    bfs((node.head,), lambda u: node.hg[u].p))

        selector = RadioGroup(
                labels=['Node %i' % i for i in range(n_nodes)], active=0,
                name='Node to inspect')
        selector.on_click(sel_node)

        plot = figure(plot_height=600, plot_width=700, y_range=(0, 20), tools=[PanTool(dimensions=['height'])])
        plot.xgrid.grid_line_color = None
        plot.xaxis.minor_tick_line_color = None
        plot.ygrid.grid_line_color = None
        plot.yaxis.minor_tick_line_color = None

        self.tr_src = ColumnDataSource(
                data={'x': [], 'y': [], 'round_color': [], 'indice_color': [],
                      'line_alpha': [], 'round': [], 'round_recv': [], 'idx': []})

        self.tr_rend = plot.circle(x='x', y='y', size=8, color='round_color',
                                   line_alpha='line_alpha', source=self.tr_src)


        self.links_src = ColumnDataSource(data={'x0': [], 'y0': [], 'x1': [],
                                                'y1': []})
        #self.links_rend = plot.add_layout(
        #        Arrow(end=NormalHead(fill_color='black'), x_start='x0', y_start='y0', x_end='x1',
        #        y_end='y1', source=self.links_src))
        self.links_rend = plot.segment(
                x0='x0', y0='y0', x1='x1',
                y1='y1', source=self.links_src)

        sel_node(0)
        curdoc().add_root(row([widgetbox(play, selector), plot], sizing_mode='fixed'))

    def extract_data(self, node, trs):
        tr_data = {'x': [], 'y': [], 'round_color': [], 'indice_color': [],
                'line_alpha': []}
        links_data = {'x0': [], 'y0': [], 'x1': [], 'y1': []}
        for u in trs:
            x = self.ids[node.hg[u].c]
            y = node.height[u]
            tr_data['x'].append(x)
            tr_data['y'].append(y)
            tr_data['round_color'].append(round_color(node.round[u]))
            tr_data['indice_color'].append(0)
            tr_data['line_alpha'].append(1 if node.famous.get(u) else 0)

            ev = node.hg[u]
            if ev.p:
                links_data['x0'].extend((x, x))
                links_data['y0'].extend((y, y))
                links_data['x1'].append(self.ids[node.hg[ev.p[0]].c])
                links_data['x1'].append(self.ids[node.hg[ev.p[1]].c])
                links_data['y1'].append(node.height[ev.p[0]])
                links_data['y1'].append(node.height[ev.p[1]])
        return tr_data, links_data

    def animate(self):
        r = randrange(len(self.main_its))
        new = next(self.main_its[r])
        if r == self.active:
            tr, links = self.extract_data(self.nodes[r], new)
            self.tr_src.stream(tr)
            self.tr_src.trigger
            self.links_src.stream(links)
        print(self.links_src.data)

App(int(sys.argv[1]))
