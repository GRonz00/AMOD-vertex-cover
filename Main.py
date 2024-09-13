import  random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.merge.util import first
from networkx import neighbors

MAX_WEIGHT = 10
MAX_NODES = 20
MIN_NODES = 5
def crate_random_graph():
    random.seed(43)
    num_nodes = random.randint(MIN_NODES, MAX_NODES)  # Random number of nodes between 5 and 10
    num_edges = random.randint(num_nodes - 1, num_nodes * (num_nodes - 1) // 2)  # Random number of edges

    # Step 2: Create a random graph
    G = nx.gnm_random_graph(num_nodes, num_edges)

    q = 0
    deg = 0
    for u in G:
        wei = random.randint(1,MAX_WEIGHT)
        G.nodes[u]['weight'] = wei
        G.nodes[u]['orig_weight'] = wei
        for v in neighbors(G,u):
            if not G[u][v]:
                G[u][v]['number'] = q
                q += 1
            deg += 1

        G.nodes[u]['deg'] = deg
        deg = 0
    return  G


if __name__ == '__main__':

    G_ori = crate_random_graph()
    G = G_ori.copy()
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig("graf")
    print(G.nodes().data('weight'))
    sol_apx = np.zeros(len(G.nodes))
    num_edge = len(G.edges())
    i = 0
    first_time = True
    while True:
        lambda_e = np.zeros(num_edge)
        R = []
        for u in G.nodes():
            R.append(u)
        #print('num_nodes='+ str(len(R))+' num_edges='+ str(len(G.edges)))

        #print(G.nodes().data('deg'))
        #print(G.edges())


        while R:
            m = MAX_WEIGHT +1
            actual_min = None
            for u in R:
                if G.nodes[u]['weight'] < m:
                    actual_min = u
                    m = G.nodes[u]['weight']
            #print('alg su nodo' + str(actual_min))
            if actual_min is not None:
                for v in nx.neighbors(G,actual_min):
                    if R.__contains__(v):
                        lambda_e[G[actual_min][v]['number']] += G.nodes[actual_min]['weight']/G.nodes[actual_min]['deg']
                        G.nodes[v]['weight'] -= lambda_e[G[actual_min][v]['number']]
                    #print('il nuovo peso di '+ str(v)+ ' è '+str(G.nodes[v]['weight']))
                R.remove(actual_min)
        #print(lambda_e)


        G_check = G.copy()
        for u in G.nodes():
            som_lambda = 0
            for v in neighbors(G,u):
                som_lambda += lambda_e[G[u][v]['number']]
            val = G.nodes[u]['orig_weight'] - som_lambda
            #print('condizione c1 per nodo '+ str(u)+ '= '+ str(val))
            if val == 0:
                sol_apx[u] = 1
                G_check.remove_node(u)
                #print('aggiunto nodo'+ str(u))
        #print('il lower bound è ='+ str(lambda_e.sum()))
        if first_time:
            lower_bound = lambda_e.sum()
            first_time = False
        if not G_check.edges():
            print('è una cover non so se minimale')
            print(sol_apx)
            for j in np.where(sol_apx == 1)[0]:
                G_check = G_ori.copy()
                node_to_remove = np.where(sol_apx == 1)[0]
                node_to_remove = np.delete(node_to_remove,np.where(node_to_remove == j))
                G_check.remove_nodes_from(node_to_remove)

                if  not G_check.edges():
                    print('ho tolto dalla soluzione '+str(j))
                    sol_apx[j] = 0
            print('la cover minimale è')
            print(sol_apx)
            break

        else:
            #print(G_check.edges())
            i += 1
            #print('iterazione ' + str(i))

            for u in G_check:
                deg = 0
                G_check.nodes[u]['weight'] = G_check.nodes[u]['orig_weight']
                for v in neighbors(G_check,u):
                    deg += 1
                G_check.nodes[u]['deg'] = deg
            G = G_check
    print('ricerca lambda eseguita ' + str(i)+ ' volte')
    sol_trovata = 0
    for u in G_ori:
        if sol_apx[u] == 1:
            sol_trovata += G_ori.nodes[u]['orig_weight']

    print('il lower bound è ' + str(lower_bound)+' mentre la loluzione trovata '+ str(sol_trovata))








