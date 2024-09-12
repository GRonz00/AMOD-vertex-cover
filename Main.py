import  random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import neighbors

if __name__ == '__main__':
    # Step 1: Generate random number of nodes and edges
    random.seed(42)
    num_nodes = random.randint(5, 10)  # Random number of nodes between 5 and 10
    num_edges = random.randint(num_nodes - 1, num_nodes * (num_nodes - 1) // 2)  # Random number of edges

    # Step 2: Create a random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)
    R = []
    q = 0
    deg = 0
    for u in G:
        wei = random.randint(1,10)
        G.nodes[u]['weight'] = wei
        G.nodes[u]['orig_weight'] = wei
        for v in neighbors(G,u):
            if not G[u][v]:
                G[u][v]['number'] = q
                q += 1
            deg += 1

        G.nodes[u]['deg'] = deg
        deg = 0
        R.append(u)
    print('num_nodes='+ str(num_nodes)+' num_edges='+ str(num_edges))
    print(G.nodes().data('weight'))
    print(G.nodes().data('deg'))
    print(G.edges())
    nx.draw(G)
    plt.savefig("graf")
    lambda_e = np.zeros(num_edges)
    while R:
        m = 11
        actual_min = None
        for u in R:
            if G.nodes[u]['weight'] < m:
                actual_min = u
                m = G.nodes[u]['weight']
        #print('alg su nodo' + str(actual_min))
        for v in nx.neighbors(G,actual_min):
            if R.__contains__(v):
                lambda_e[G[actual_min][v]['number']] += G.nodes[actual_min]['weight']/G.nodes[actual_min]['deg']
                G.nodes[v]['weight'] -= lambda_e[G[actual_min][v]['number']]
            #print('il nuovo peso di '+ str(v)+ ' è '+str(G.nodes[v]['weight']))
        R.remove(actual_min)
    print(lambda_e)
    sol_apx = np.zeros(num_nodes)
    for u in G:
        som_lambda = 0
        for v in neighbors(G,u):
            som_lambda += lambda_e[G[u][v]['number']]
        val = G.nodes[u]['orig_weight'] - som_lambda
        print('condizione c1 per nodo '+ str(u)+ '= '+ str(val))
        if val == 0:
            sol_apx[u] = 1
    print('il lower bound è ='+ str(lambda_e.sum()))








