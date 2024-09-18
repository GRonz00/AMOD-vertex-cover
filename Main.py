import random
import time

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gurobipy import GRB

MAX_WEIGHT = 100
MAX_NODES = 1100
MIN_NODES = 50
DEBUGGING = False
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

def solve_weighted_vertex_cover(G):
    start_time_gurobi = time.time()
    # Crea un modello Gurobi
    model = gp.Model("WeightedVertexCover")

    # Disabilita l'output di log
    model.setParam('LogToConsole', 0)

    # Crea variabili per ogni nodo: 1 se è incluso nel vertex cover, 0 altrimenti
    x = model.addVars(G.nodes(), vtype=GRB.BINARY, name="x")

    # Aggiungi vincoli per ogni arco
    for u, v in G.edges():
        model.addConstr(x[u] + x[v] >= 1, f"cover_{u}_{v}")

    # Obiettivo: minimizzare il peso totale dei nodi nel vertex cover
    model.setObjective(gp.quicksum(G.nodes[n]['weight'] * x[n] for n in G.nodes()), GRB.MINIMIZE)

    # Risolvi il modello
    model.optimize()

    # Estrai la soluzione
    val_vertex_cover = sum(G.nodes[n]['weight'] for n in G.nodes() if x[n].x > 0.5)
    

    return val_vertex_cover, time.time()-start_time_gurobi

if __name__ == '__main__':
    random.seed(46)
    confronto_sol_list =[]
    confronto_gurobi_list = []
    confronto_tempo_list = []
    confronto_gurobi_LB = []
    sol_uguale_LB= 0
    sol_esatta = 0
    timeout_seconds = 3 * 60
    file = open("risultati.txt","w")
    file.close()
    for z in range(3):
        time_sol = 0
        sol_gurobi = []
        little_inst = False
        exceeded_time = False
        G_ori = crate_random_graph()
        G = G_ori.copy()
        if len(G.nodes) < 400:
            little_inst = True
            sol_gurobi = solve_weighted_vertex_cover(G)
        sol_apx = np.zeros(len(G.nodes))
        num_edge = len(G.edges())
        #i = 1
        first_time = True
        lower_bound = 0
        start_time = time.time()
        file_uno = open("risultati.txt","a")
        file_uno.write('grafico num' + str(z) + 'num_nodes='+ str(len(G.nodes()))+' num_edges='+ str(len(G.edges))+"\n")
        if DEBUGGING:
            print('grafico num' + str(z) + 'num_nodes='+ str(len(G.nodes()))+' num_edges='+ str(len(G.edges)))
        while True:

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                exceeded_time = True
                file_uno.write(f"Timeout raggiunto dopo {timeout_seconds/60} minuti."+"\n\n")
                if DEBUGGING:
                    print(f"Timeout raggiunto dopo {timeout_seconds/60} minuti.")
                break

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
                if DEBUGGING:
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
                time_sol = round(time.time()-start_time,2)
                break

            else:
                #i += 1
                for u in G_check:
                    G_check.nodes[u]['weight'] = G_check.nodes[u]['orig_weight']
                G = G_check

        if not exceeded_time:
            #print('ricerca lambda eseguita ' + str(i)+ ' volte')
            valore_soluzione = sum(G_ori.nodes[u]['orig_weight'] for u in G_ori.nodes if sol_apx[u] == 1)
            confronto_sol = round((valore_soluzione - lower_bound) / lower_bound * 100, 2)
            if confronto_sol == 0:
                sol_uguale_LB += 1
            file_uno.write('il lower bound è ' + str(lower_bound)+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi differenza in percentuale =' + str(confronto_sol)+'%\n')
            if not little_inst:
                file_uno.write('soluzione trovata in '+str(round(time_sol,2))+' secondi\n\n')
            if DEBUGGING:
                print('il lower bound è ' + str(lower_bound)+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi differenza in percentuale =' + str(confronto_sol)+'%')
                print('soluzione trovata in '+str(round(time_sol,2))+' secondi')
            if little_inst:
                confronto_val_gurobi = round((valore_soluzione - sol_gurobi[0]) / sol_gurobi[0] * 100, 2)

                confronto_gurobi_list.append(confronto_val_gurobi)
                confronto_tempo_list.append(sol_gurobi[1]-time_sol)
                confronto_gurobi_LB.append(round(sol_gurobi[0]-lower_bound/lower_bound*100,2))
                file_uno.write('la soluzione trovata con gurobi ' + str(sol_gurobi[0])+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi aumento in percentuale =' + str(confronto_val_gurobi)+'%\n')
                file_uno.write('tempo gurobi '+ str(round(sol_gurobi[1],2))+' s mentre tempo soluzione approssimata '+ str(time_sol)+ ' quindi aumento in percentuale '+ str(sol_gurobi[1]-time_sol)+'%\n\n')
                if DEBUGGING:
                    print('la soluzione trovata con gurobi ' + str(sol_gurobi[0])+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi aumento in percentuale =' + str(confronto_val_gurobi)+'%')
                    print('tempo gurobi '+ str(round(sol_gurobi[1],2))+' s mentre tempo soluzione approssimata '+ str(time_sol)+ ' quindi differenza '+ str(sol_gurobi[1]-time_sol)+'%')

                if confronto_val_gurobi <= 0:
                    sol_esatta +=1
            else:
                confronto_gurobi_LB.append(-1)
                if confronto_sol == 0:
                    sol_esatta += 1
            confronto_sol_list.append(confronto_sol)
        file_uno.close()
    file_uno = open("risultati.txt","a")
    file_uno.write('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_sol_list),2))+'% rispetto al lower bound e per '+ str(sol_uguale_LB)+' volte è stato trovato valore soluzione uguale al lower bound\n')
    file_uno.write('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_gurobi_list),2))+'% rispetto alla soluzione di gurobi e per '+ str(sol_esatta)+' volte è stata trovata la soluzione esatta\n')
    file_uno.write('tempo medio impiegato in più per trovare la soluzione '+ str(round(np.mean(confronto_tempo_list),2)) + 's')
    file_uno.close()
    print('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_sol_list),2))+'% rispetto al lower bound e per '+ str(sol_uguale_LB)+' volte è stato trovato valore soluzione uguale al lower bound')
    print('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_gurobi_list),2))+'% rispetto alla soluzione di gurobi e per '+ str(sol_esatta)+' volte è stata trovata la soluzione esatta')
    print('tempo medio impiegato in più per trovare la soluzione '+ str(round(np.mean(confronto_tempo_list),2)) + 's')









