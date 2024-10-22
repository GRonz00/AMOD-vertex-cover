import random
import time

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gurobipy import GRB
N_ISTANZE = 80
LITTLE_IST = 500
MAX_WEIGHT = 100
MAX_NODES = 300
MIN_NODES = 100
DEBUGGING = False
def crate_random_graph(z):
    num_nodes = random.randint(MIN_NODES + 200*z, MAX_NODES+ 200*z)
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
    model.setParam('TimeLimit', 8*60)
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
    # Controlla lo stato del modello dopo l'ottimizzazione
    status = model.status
    val_vertex_cover = -1
    # Stampa lo stato dell'ottimizzazione
    if status == GRB.OPTIMAL:
        print("Soluzione ottima trovata")
        val_vertex_cover = sum(G.nodes[n]['weight'] for n in G.nodes() if x[n].x > 0.5)
    elif status == GRB.INFEASIBLE:
        print("Il modello non ha soluzioni fattibili")
    elif status == GRB.UNBOUNDED:
        print("Il modello è illimitato (unbounded)")
    elif status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Se c'è una soluzione, estrai il miglior vertex cover trovato
            val_vertex_cover = sum(G.nodes[n]['weight'] for n in G.nodes() if x[n].x > 0.5)
            print("Soluzione approssimata trovata")
    else:
        print(f"Altro stato del modello: {status}")
    # Estrai la soluzione
    return val_vertex_cover, time.time()-start_time_gurobi

if __name__ == '__main__':
    random.seed(42)
    confronto_sol_list =[]
    confronto_gurobi_list = []
    tempo_gurobi_list = []
    tempo_sol_list = []
    confronto_gurobi_LB = []
    num_nodes = []
    sol_uguale_LB= 0
    sol_esatta = 0

    timeout_seconds = 15 * 60
    file = open("risultati.txt","w")
    file.close()
    for z in range(N_ISTANZE):
        print(z)
        time_sol = 0
        sol_gurobi = []
        little_inst = False
        exceeded_time = False
        G_ori = crate_random_graph(z // 20)
        G = G_ori.copy()
        file_uno = open("risultati.txt","a")
        file_uno.write('grafico num' + str(z) + 'num_nodes='+ str(len(G.nodes()))+' num_edges='+ str(len(G.edges))+"\n")
        file_uno.close()
        file_uno = open("risultati.txt","a")
        num_nodes.append(len(G.nodes))
        if len(G.nodes) < LITTLE_IST:
            little_inst = True
            sol_gurobi = solve_weighted_vertex_cover(G)
        sol_apx = np.zeros(len(G.nodes))
        num_edge = len(G.edges())
        first_time = True
        lower_bound = 0
        start_time = time.time()
        if DEBUGGING:
            print('grafico num' + str(z) + ' num_nodes='+ str(len(G.nodes()))+' num_edges='+ str(len(G.edges)))
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
                node_weights = {node: G_ori.nodes[node]['orig_weight'] for node in G_ori.nodes}
                nodes_ordered_by_weight = sorted(np.where(sol_apx == 1)[0], key=lambda x: node_weights[x], reverse=True)
                for j in nodes_ordered_by_weight:
                    G_check = G_ori.copy()
                    node_to_remove = np.delete(np.where(sol_apx == 1)[0], np.where(np.where(sol_apx == 1)[0] == j))
                    G_check.remove_nodes_from(node_to_remove)

                    if  not G_check.edges():
                        #print('ho tolto dalla soluzione '+str(j))
                        sol_apx[j] = 0
                #print('la cover minimale è' + str(np.where(sol_apx == 1)[0]))
                time_sol = round(time.time()-start_time,2)
                tempo_sol_list.append(time_sol)
                break

            else:
                #i += 1
                for u in G_check:
                    G_check.nodes[u]['weight'] = G_check.nodes[u]['orig_weight']
                G = G_check

        if not exceeded_time:
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
                if sol_gurobi[0] != -1:
                    confronto_val_gurobi = round((valore_soluzione - sol_gurobi[0]) / sol_gurobi[0] * 100, 2)
                    confronto_gurobi_list.append(confronto_val_gurobi)
                    if confronto_val_gurobi <= 0:
                        sol_esatta +=1
                tempo_gurobi_list.append(sol_gurobi[1])
                confronto_gurobi_LB.append(round((sol_gurobi[0]-lower_bound)/lower_bound*100,2))
                file_uno.write('la soluzione trovata con gurobi ' + str(sol_gurobi[0])+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi aumento in percentuale =' + str(confronto_val_gurobi)+'%\n')
                file_uno.write('tempo gurobi '+ str(round(sol_gurobi[1],2))+' s mentre tempo soluzione approssimata '+ str(time_sol) + 's\n\n')
                if DEBUGGING:
                    print('la soluzione trovata con gurobi ' + str(sol_gurobi[0])+' mentre il valore della soluzione trovata '+ str(valore_soluzione) + ' quindi aumento in percentuale =' + str(confronto_val_gurobi)+'%')
                    print('tempo gurobi '+ str(round(sol_gurobi[1],2))+' s mentre tempo soluzione approssimata '+ str(time_sol)+ ' quindi differenza '+ str(sol_gurobi[1]-time_sol)+'%')


            else:
                if confronto_sol == 0:
                    sol_esatta += 1
            confronto_sol_list.append(confronto_sol)
        file_uno.close()
    #metti risultati finali su file
    file_uno = open("risultati.txt","a")
    file_uno.write('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_sol_list),2))+'% rispetto al lower bound e per '+ str(sol_uguale_LB)+' volte è stato trovato valore soluzione uguale al lower bound\n')
    file_uno.write('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_gurobi_list),2))+'% rispetto alla soluzione di gurobi e per '+ str(sol_esatta)+' volte è stata trovata la soluzione esatta\n')
    file_uno.write('tempo medio impiegato in più per trovare la soluzione  su istanze piccole '+ str(round(np.mean([a - b for a, b in zip(tempo_sol_list, tempo_gurobi_list)if b != -1]),2)) + 's')
    file_uno.close()
    #crea grafici
    for j in range(2):

        # Imposta la posizione delle barre
        indices = np.arange(len(confronto_sol_list[j*(N_ISTANZE//4):(j+1)*N_ISTANZE//4]))
        plt.figure()
        # Grafico dei tempi
        plt.bar(indices, tempo_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione algoritmo', color='blue', alpha=0.7)
        plt.bar(indices, tempo_gurobi_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione Gurobi', color='red', alpha=0.7)
        plt.axhline(y=480, color='r', linestyle='--', label='Timeout Gurobi')

        # Aggiungi etichette e titolo
        plt.xlabel('Istanza')
        plt.ylabel('Tempo')
        plt.title('Confronto Tempo Soluzioni')
        plt.xticks(ticks=indices, labels=[str(i) for i in num_nodes[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]], rotation=90)
        plt.legend()
        plt.ylim(0, max(max(tempo_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]),max(tempo_gurobi_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]) * 1.2))


        plt.savefig('gTempo'+str(j))

        plt.figure()
        plt.bar(indices, confronto_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione algoritmo', color='blue', alpha=0.7)
        plt.bar(indices, confronto_gurobi_LB[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione Gurobi', color='red', alpha=0.7)
        plt.xlabel('Istanza')
        plt.ylabel('Percentuale')
        plt.title('Confronto Valori soluzioni')
        plt.xticks(ticks=indices, labels=[str(i) for i in num_nodes[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]], rotation=90)
        plt.legend()
        plt.ylim(0, max(confronto_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]) * 1.2)
        plt.savefig('gValori'+ str(j))
    for j in range(2,4):

        # Imposta la posizione delle barre
        indices = np.arange(len(confronto_sol_list[j*(N_ISTANZE//4):(j+1)*N_ISTANZE//4]))
        plt.figure()
        # Grafico dei tempi
        plt.bar(indices, tempo_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione algoritmo', color='blue', alpha=0.7)

        # Aggiungi etichette e titolo
        plt.xlabel('Istanza')
        plt.ylabel('Tempo')
        plt.title('Tempo Soluzioni Grandi Istanze')
        plt.xticks(ticks=indices, labels=[str(i) for i in num_nodes[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]], rotation=90)
        plt.legend()
        plt.ylim(0, max(tempo_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]) * 1.2)

        plt.savefig('gTempo'+str(j))

        plt.figure()
        plt.bar(indices, confronto_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4], width=0.4, label='Soluzione algoritmo', color='blue', alpha=0.7)
        plt.xlabel('Istanza')
        plt.ylabel('Percentuale')
        plt.title('Confronto Valori soluzioni grandi istanze')
        plt.xticks(ticks=indices, labels=[str(i) for i in num_nodes[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]], rotation=90)
        plt.legend()
        plt.ylim(0, max(confronto_sol_list[j*N_ISTANZE//4:(j+1)*N_ISTANZE//4]) * 1.2)
        plt.savefig('gValori'+ str(j))
    print('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_sol_list),2))+'% rispetto al lower bound e per '+ str(sol_uguale_LB)+' volte è stato trovato valore soluzione uguale al lower bound')
    print('in media la soluzione trovata è stata più grande del '+ str(round(np.mean(confronto_gurobi_list),2))+'% rispetto alla soluzione di gurobi e per '+ str(sol_esatta)+' volte è stata trovata la soluzione esatta')
    print('tempo medio impiegato in più per trovare la soluzione '+ str(round(np.mean([a - b for a, b in zip(tempo_sol_list, tempo_gurobi_list)if b != -1]),2)) + 's')









