import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import neighbors

MAX_WEIGHT = 100
MAX_NODES = 50
MIN_NODES = 5
def crate_random_graph():
    num_nodes = random.randint(MIN_NODES, MAX_NODES)
    num_edges = random.randint(num_nodes - 1, num_nodes * (num_nodes - 1) // 2)
    G = nx.gnm_random_graph(num_nodes, num_edges)

    q = 0 #do un nome all'arco
    for u, v in G.edges:
        G[u][v]['number'] = q
        q += 1

    # Imposta i pesi e numeri degli archi
    for u in G.nodes:
        wei = random.randint(1, MAX_WEIGHT)
        G.nodes[u]['weight'] = wei
        G.nodes[u]['orig_weight'] = wei
    return  G

def save_graf(G):
    labels = {}
    pos = nx.spring_layout(G, seed=3113794652)
    for u in G.nodes():
        labels[u] = str(u)+ '/' + str(G.nodes[u]['weight'])
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.savefig("graf"+ str(z))
    print(G.nodes().data('weight'))

if __name__ == '__main__':
    random.seed(46)
    confronto_sol_list =[]
    sol_esatta= 0
    for z in range(100):
        G_ori = crate_random_graph()
        G = G_ori.copy()
        sol_apx = np.zeros(len(G.nodes))
        num_edge = len(G.edges())
        #i = 1
        first_time = True
        lower_bound = 0
        print('grafico num' + str(z) + 'num_nodes='+ str(len(G.nodes()))+' num_edges='+ str(len(G.edges)))
        while True:
            lambda_e = np.zeros(num_edge)
            R = set(G.nodes)
            while R:
                actual_min = min(R, key=lambda u: G.nodes[u]['weight'])
                prop_weight = sum(G.nodes[v]['weight'] for v in G.neighbors(actual_min) if v in R)

                for v in G.neighbors(actual_min):
                    if v in R:
                        #alpha = 1 / deg(u)
                        #lambda_e[G[actual_min][v]['number']] += G.nodes[actual_min]['weight']/G.degree[actual_min]
                        lambda_e[G[actual_min][v]['number']] += G.nodes[actual_min]['weight'] * G.nodes[v]['weight'] / prop_weight
                        G.nodes[v]['weight'] -= lambda_e[G[actual_min][v]['number']]
                R.remove(actual_min)
            #print(lambda_e)
            G_check = G.copy()
            temp_node = None
            min_val = MAX_WEIGHT
            for u in G.nodes():
                som_lambda = sum(lambda_e[G[u][v]['number']] for v in G.neighbors(u))
                val = G.nodes[u]['orig_weight'] - som_lambda
                #print('condizione c1 per nodo '+ str(u)+ '= '+ str(val))
                if val < min_val:
                    temp_node = u
                    min_val = val
                if val == 0:
                    sol_apx[u] = 1
                    G_check.remove_node(u)
                    #print('aggiunto nodo'+ str(u))
            #print('il lower bound è ='+ str(lambda_e.sum()))
            if  np.all(sol_apx[np.array(list(G.nodes()))] != 1):
                print('la ricerca di lambda non è riuscita a potare nessuna c1 =0')
                #in caso non riesco a soddisfare c1 uso quello che più ci si avvicina
                sol_apx[temp_node] = 1
                G_check.remove_node(temp_node)
            if first_time:
                lower_bound = round(lambda_e.sum(),2)
                first_time = False
            if not G_check.edges():
                #print('è una cover non so se minimale'+ str(np.where(sol_apx == 1)[0]))
                for j in np.where(sol_apx == 1)[0]:
                    G_check = G_ori.copy()
                    node_to_remove = np.delete(np.where(sol_apx == 1)[0], np.where(np.where(sol_apx == 1)[0] == j))
                    G_check.remove_nodes_from(node_to_remove)

                    if  not G_check.edges():
                        #print('ho tolto dalla soluzione '+str(j))
                        sol_apx[j] = 0
                #print('la cover minimale è' + str(np.where(sol_apx == 1)[0]))
                break

            else:
                #i += 1
                for u in G_check:
                    G_check.nodes[u]['weight'] = G_check.nodes[u]['orig_weight']
                G = G_check
        #print('ricerca lambda eseguita ' + str(i)+ ' volte')
        valore_soluzione = sum(G_ori.nodes[u]['orig_weight'] for u in G_ori.nodes if sol_apx[u] == 1)
        confronto_sol = round((valore_soluzione - lower_bound) / (lower_bound) * 100,2)
        if confronto_sol == 0:
            sol_esatta += 1

        print('il lower bound è ' + str(lower_bound)+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi differenza in percentuale =' + str(confronto_sol)+'%')
        confronto_sol_list.append(confronto_sol)
    print('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_sol_list),2))+'% rispetto al lower bound e per '+ str(sol_esatta)+' volte è stata trovata la soluzione ottima')








