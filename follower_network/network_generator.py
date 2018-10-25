import networkx as nx
import traceback
import logging as logger
from util import globals
import matplotlib.pyplot as plt


def plot(G):
    nx.draw(G)

    plt.show()


def main():
    try:
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
        filename_read = "F:/tmp/edges.txt"
        G = nx.Graph()

        all_edges = [line.rstrip('\n') for line in open(filename_read)]

        for edge in all_edges:
            edges = edge.split(",")
            fr = edges[0]
            to = edges[1]
            G.add_edge(fr, to)

        print("nb of nodes:" + str(G.number_of_nodes()))
        print("nb of edges:" + str(G.number_of_edges()))
        print("plotting started")
        #plot(G)
        nx.draw(G)
        plt.savefig("F:/tmp/path.png")
        print("plotting completed")

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()