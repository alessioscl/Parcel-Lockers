import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.compose import ColumnTransformer
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from collections import defaultdict
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def prepare_data(n_days, n_requests_per_day, n_lockers):
    """
    Genera i dataset di input per il problema Dynamic Parcel Locker Allocation
    e li converte in strutture compatibili con Gurobi.
    
    Parametri:
      - n_days: numero di giorni (time horizon T)
      - n_requests_per_day: richieste per ciascun giorno
      - n_lockers: numero di locker (insieme J)
      
    Restituisce un dizionario contenente:
      - 'df_requests': DataFrame con i dati delle richieste, con colonne:
          i, x, y, type, weight, pickup_time, arrival_day, size, tau (colonna aggiuntiva)
      - 'df_lockers': DataFrame con i dati dei locker, con colonne:
          j, x, y, C_1, C_2, C_3
      - 'phi': DataFrame (matrice) di compatibilità (ϕ_ij) tra richieste e locker
      - 'T': insieme dei giorni
      - 'I': lista degli ID delle richieste
      - 'J': lista degli ID dei locker
      - 'K': insieme delle tipologie di compartimenti (es. {1, 2, 3})
      - 'requests': dizionario con le informazioni di ciascuna richiesta (indicizzato per i)
      - 'lockers': dizionario con le informazioni di ciascun locker (indicizzato per j)
      - 'phi_dict': dizionario con chiave (i, j) e valore 0 o 1, per la compatibilità
      - 'C': dizionario con chiave (j, k) e valore numero di compartimenti di tipo k nel locker j
    """
    # -------------------------------
    # Generazione dati richieste (df_requests)
    # -------------------------------
    T = set(range(n_days))
    requests_per_day = {}
    total_requests = 0
    # Per ogni giorno genera un numero di richieste variabile (base ±10)
    for t in sorted(T):
        # Assicuriamo un minimo di 1 richiesta per giorno
        arrivals = np.random.randint(max(n_requests_per_day - 10, 1), n_requests_per_day + 11)
        requests_per_day[t] = arrivals
        total_requests += arrivals
    I = list(range(1, total_requests + 1))
    I_x = {i: np.random.randint(0, 100) for i in I}
    I_y = {i: np.random.randint(0, 100) for i in I}
    I_type = {i: np.random.choice(['premium', 'standard'], p=[0.7, 0.3]) for i in I}
    I_w = {i: 5 if I_type[i] == 'premium' else 1 for i in I}
    I_alpha = {i: np.random.choice([1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.2]) for i in I}
    I_size = {i: np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]) for i in I}  # s_i

    # Assegna ad ogni richiesta il giorno di arrivo (τ_i)
    I_t = {}
    counter = 1
    for t in sorted(T):
        for i in range(counter, counter + requests_per_day[t]):
            I_t[i] = t
        counter += requests_per_day[t]

    df_requests = pd.DataFrame({
        "i": I,
        "x": [I_x[i] for i in I],
        "y": [I_y[i] for i in I],
        "type": [I_type[i] for i in I],
        "weight": [I_w[i] for i in I],
        "pickup_time": [I_alpha[i] for i in I],
        "arrival_day": [I_t[i] for i in I],
        "size": [I_size[i] for i in I]
    })
    
    df_requests['total_requests_day'] = df_requests['arrival_day'].apply(lambda t: requests_per_day[t])
    df_requests['order'] = df_requests.groupby('arrival_day').cumcount() + 1
    df_requests['order_ss'] = df_requests.groupby(['arrival_day','size']).cumcount() + 1
    df_requests['order_st'] = df_requests.groupby(['arrival_day','type']).cumcount() + 1
    # -------------------------------
    # Generazione dati locker (df_lockers)
    # -------------------------------
    J = list(range(1, n_lockers + 1))
    
    # Se n_lockers == 5 usiamo posizioni fisse, altrimenti generiamo coordinate casuali
    #if n_lockers == 5:
    #    J_x = {1: 50, 2: 25, 3: 25, 4: 75, 5: 75}
    #    J_y = {1: 50, 2: 25, 3: 75, 4: 25, 5: 75}
    #else:
    J_x = {j: np.random.randint(0, 100) for j in J}
    J_y = {j: np.random.randint(0, 100) for j in J}
    
    # Definizione dei compartimenti per ciascun locker
    J_C_1 = {j: 10 for j in J}  # Compartimenti piccoli
    J_C_2 = {j: 10 for j in J}  # Compartimenti medi
    J_C_3 = {j: 5  for j in J}  # Compartimenti grandi
    
    df_lockers = pd.DataFrame({
        "j": J,
        "x": [J_x[j] for j in J],
        "y": [J_y[j] for j in J],
        "C_1": [J_C_1[j] for j in J],
        "C_2": [J_C_2[j] for j in J],
        "C_3": [J_C_3[j] for j in J]
    })
    
    # -------------------------------
    # Matrice di compatibilità phi: ϕ_ij = 1 se la distanza tra richiesta i e locker j è < 50
    # -------------------------------
    phi = pd.DataFrame(index=df_requests['i'], columns=df_lockers['j'], dtype=int)
    for _, row_req in df_requests.iterrows():
        for _, row_lock in df_lockers.iterrows():
            dist = np.sqrt((row_req['x'] - row_lock['x'])**2 + (row_req['y'] - row_lock['y'])**2)
            phi.at[row_req['i'], row_lock['j']] = 1 if dist < 50 else 0

    # -------------------------------
    # Preparazione dei dati per Gurobi
    # -------------------------------
    # Dizionari per richieste e locker
    requests_dict = df_requests.set_index('i').to_dict(orient='index')
    lockers_dict = df_lockers.set_index('j').to_dict(orient='index')
    
    phi_dict = {(i, j): int(phi.loc[i, j]) for i in I for j in J}
    
    
    K = {1, 2, 3}
    
    C = {}
    for j in J:
        for k in K:
            C[(j, k)] = df_lockers.loc[df_lockers['j'] == j, f'C_{k}'].iloc[0]
    

    # -------------------------------
    # Restituisce tutti i dati
    # -------------------------------
    return {
        "df_requests": df_requests,
        "df_lockers": df_lockers,
        "phi": phi,
        "T": T,
        "I": I,
        "J": J,
        "K": K,
        "requests": requests_dict,
        "lockers": lockers_dict,
        "phi_dict": phi_dict,
        "C": C
    }

def read_from_excel(instance_file, locker_file, n_lockers):
    """
    Reads data from Excel files and formats it to match prepare_data() output structure.
    
    Parameters:
    - instance_file: path to Excel file containing requests data
    - locker_file: path to Excel file containing locker data
    - n_lockers: number of lockers to use (4, 6, or 8)
    
    Returns:
    - Dictionary with same structure as prepare_data() output
    """
    # Read requests data
    df_requests = pd.read_excel(instance_file)
    #df_requests = modify_pickup_to_4_days(df_requests)
    #mapping = {
    #    1:1,
    #    2:1,
    #    3:1
    #}
    #df_requests['pickup_time'] = df_requests['pickup_time'].map(mapping)

    # Read lockers data
    df_lockers = pd.read_excel(locker_file)
    
    T = set(df_requests['arrival_day'].unique())
    I = list(df_requests['i'].unique())
    J = list(df_lockers['j'].unique())
    K = {1, 2, 3}
    
    # Create requests dictionary
    requests_dict = df_requests.set_index('i').to_dict(orient='index')
    lockers_dict = df_lockers.set_index('j').to_dict(orient='index')
    
    # Create phi matrix and dictionary
    distance = pd.DataFrame(index=df_requests['i'], columns=df_lockers['j'], dtype=int)
    phi = pd.DataFrame(index=df_requests['i'], columns=df_lockers['j'], dtype=int)
    for _, row_req in df_requests.iterrows():
        for _, row_lock in df_lockers.iterrows():
            dist = np.sqrt((row_req['x'] - row_lock['x'])**2 + (row_req['y'] - row_lock['y'])**2)
            distance.at[row_req['i'], row_lock['j']] = dist
            phi.at[row_req['i'], row_lock['j']] = 1 if dist < 50 else 0
    phi_dict = {(i, j): int(phi.loc[i, j]) for i in I for j in J}

    # Create capacity dictionary
    C = {}
    for j in J:
        for k in K:
            C[(j, k)] = df_lockers.loc[df_lockers['j'] == j, f'C_{k}'].iloc[0]
    
    # Return formatted data
    return {
        "df_requests": df_requests,
        "df_lockers": df_lockers,
        "phi": phi,
        "T": T,
        "I": I,
        "J": J,
        "K": K,
        "requests": requests_dict,
        "lockers": lockers_dict,
        "phi_dict": phi_dict,
        "C": C,
        "distance": distance
    }

def sample_requests_per_day(data, n_samples_per_day=10):
    # Filter out P7 records before sampling
    filtered_df = data[~data['policy'].str.contains('P7')]
    
    sampled_requests = (
        filtered_df.groupby('arrival_day')
        .apply(lambda group: group.sample(n=min(n_samples_per_day, len(group)), random_state=42))
        .reset_index(drop=True)
    )
    return sampled_requests

def oracle(data):
    T = data["T"]
    I = data["I"]
    J = data["J"]
    K = data["K"]
    requests = data["requests"]
    lockers = data["lockers"]
    phi_dict = data["phi_dict"]
    C = data["C"]

    max_pickup = max(requests[i]['pickup_time'] for i in I)
    n_days = len(T)
    T_ext = set(range(n_days + max_pickup + 1))

    # Creazione del modello Gurobi
    model = gb.Model('ParcelLockerOracle')

    # Definizione delle variabili (definite come globali per essere usate in extract_solution)
    global x, y, z, u
    x = model.addVars(I, vtype=GRB.BINARY, name='x')
    y = model.addVars(I, J, vtype=GRB.BINARY, name='y')
    z = model.addVars(I, J, K, vtype=GRB.BINARY, name='z')
    u = model.addVars(I, J, K, T_ext, vtype=GRB.BINARY, name='u')

    # CONSTRAINTS
    # (1) Una richiesta viene assegnata ad uno e un solo locker se è accettata
    for i in I:
        model.addConstr(gb.quicksum(y[i, j] for j in J) == x[i])
    
    # (2) Solo i locker compatibili possono essere assegnati
    for i in I:
        for j in J:
            if phi_dict[(i, j)] == 0:
                model.addConstr(y[i, j] == 0)
    
    # (3) Per ciascuna richiesta, assegnazione del compartimento di tipo sufficiente
    for i in I:
        for j in J:
            model.addConstr(gb.quicksum(z[i, j, k] for k in K if k >= requests[i]['size']) == y[i, j])
    
    # (4) Relazione tra z e u: per ogni richiesta, il compartimento resta occupato per t = 1..pickup_time
    for i in I:
        arrival_day = requests[i]['arrival_day']
        pickup_time = requests[i]['pickup_time']
        for j in J:
            for k in K:
                for t in range(1, pickup_time + 1):
                    t_index = arrival_day + t
                    model.addConstr(u[i, j, k, t_index] == z[i, j, k])
    
    # (5) Capacità dei locker: ad ogni istante t, il numero di compartimenti occupati non supera la capacità
    for t in T_ext:
        for j in J:
            for k in K:
                model.addConstr(gb.quicksum(u[i, j, k, t] for i in I) <= C[(j, k)])
    
    # OBJECTIVE FUNCTION: massimizzare la somma dei pesi delle richieste assegnate
    model.setObjective(gb.quicksum(requests[i]['weight'] * x[i] for i in I), GRB.MAXIMIZE)
    model.update()
    # Facoltativo: model.write(f'model_{n_lockers}_lockers.lp')
    model.setParam('OutputFlag', 0) 
    model.optimize()
    return(model)

def extract_solution(data):
    """
    Estrae la soluzione ottimale dal modello Gurobi e calcola per ogni richiesta
    le seguenti feature:
      - arrival_day
      - type
      - size
      - numero di compatibili per ogni richiesta 
      - capacità residua dei compatibili al arrival_day (somma box liberi e box che si liberano domani)
      - expected future request (richieste attese nella giornata - richieste già arrivate)
      - expected future request same size
      - expected future request type
      - capacità sicuramente occupata domani (richieste assegnate a quel locker oggi fino al momento di arrivo della richiesta)
      - POLICY: - P1: locker con massima capacità libera
                - P2: locker con minima capacità libera
                - P3: locker più centrale (più vicino a (50,50))
                - P4: locker più periferico (più lontano da (50,50))
                - P5: altro
                - P6: non assegnato
                - P7: unico assegnamento
                - P8: locker più utilizzato durante il giorno
                - P9: locker meno utilizzato durante il giorno
    La funzione utilizza le variabili Gurobi (x, y, z) definite globalmente e i dati 
    contenuti nel dizionario "data" (output di prepare_data).
    
    La funzione deve essere chiamata dopo che il modello è stato ottimizzato.
    """
    


    # Recupero dataframe e dizionari di input
    df_requests = data["df_requests"].copy()
    requests = data["requests"]
    phi_dict = data["phi_dict"]
    C = data["C"]
    J = data["J"]
    K = data["K"]
    lockers = data["lockers"]  # contiene coordinate (x,y) per ogni locker
    
    # Numero di giorni e richieste per giorno (assumiamo costante)
    n_days = len(data["T"])
    n_requests_per_day = int(df_requests.groupby("arrival_day").size().iloc[0])
    max_pickup = max(req['pickup_time'] for req in requests.values())
    

    max_day = n_days + max_pickup + 2  # per coprire eventuali giorni oltre il time horizon
    occupancy = {(j, k, d): 0 for j in J for k in K for d in range(max_day)}
    freeing = {(j, k, d): 0 for j in J for k in K for d in range(max_day)}
    

    sorted_requests = sorted(data["I"], key=lambda i: (requests[i]['arrival_day'], requests[i].get('order', i)))
    
    # Tracciamo le richieste assegnate fino ad un certo momento per ogni giorno
    assigned_until_now = {d: {} for d in range(n_days+1)}  # giorno -> {i -> (j, k)}
    assignment = {}
    # Per ogni richiesta assegnata, ricaviamo locker e compartimento utilizzato e aggiorniamo occupancy e freeing.
    for i in sorted_requests:
            d = requests[i]['arrival_day']
            p = requests[i]['pickup_time']
            assigned_j = None
            for j in J:
                if y[i, j].X > 0.5:
                    assigned_j = j
                    break
            assignment[i] = assigned_j  # memorizziamo l'assegnazione (anche se None)
            if assigned_j is None:
                continue
            # Individuiamo il compartimento (tipo) assegnato (deve essercene uno solo)
            assigned_k = None
            for k in K:
                if z[i, assigned_j, k].X > 0.5:
                    assigned_k = k
                    break
            if assigned_k is None:
                continue
            
            #assignment[i] = (assigned_j, assigned_k)

            # Memorizziamo l'assegnazione per questa richiesta
            if d not in assigned_until_now:
                assigned_until_now[d] = {}
            assigned_until_now[d][i] = (assigned_j, assigned_k)
            
            # La richiesta occupa il compartimento dal giorno (d+1) fino a (d+p)
            for day in range(d + 1, d + p + 1):
                occupancy[(assigned_j, assigned_k, day)] += 1
            
            # Il compartimento si libera il giorno (d + p + 1)
            free_day = d + p + 1
            if free_day < max_day:
                freeing[(assigned_j, assigned_k, free_day)] += 1
                
    
    # Calcolo delle expected future request:
    df_requests['order'] = df_requests.groupby('arrival_day').cumcount() + 1
    
    # Definizione delle probabilità di size e type
    size_probs = {1: 0.5, 2: 0.3, 3: 0.2}
    type_probs = {'premium': 0.7, 'standard': 0.3}
    
    # Calcolo delle expected future requests basato sulle probabilità
    df_requests['expected_future'] = 100 - df_requests['order']


    df_requests['order_ss'] = df_requests.groupby(['arrival_day','size']).cumcount() + 1
    df_requests['order_st'] = df_requests.groupby(['arrival_day','type']).cumcount() + 1
    # Aggiunta dei campi per expected_future_same_size e expected_future_type
    df_requests['expected_future_same_size'] = 0
    df_requests['expected_future_type'] = 0
    
    # Calcolo dei valori attesi per ogni richiesta
    for idx, row in df_requests.iterrows():
        #remaining_requests = 100 - row['order']
        
        # Calcolo delle richieste attese della stessa dimensione
        expected_same_size = 100 * size_probs[row['size']]
        df_requests.loc[idx, 'expected_future_same_size'] = expected_same_size - row['order_ss']
        
        # Calcolo delle richieste attese dello stesso tipo
        expected_same_type = 100 * type_probs[row['type']]
        df_requests.loc[idx, 'expected_future_type'] = expected_same_type - row['order_st']

    # Ordinamento delle richieste per giorno e ordine di arrivo
    df_requests.sort_values(by=['arrival_day', 'order'], inplace=True)
    
    # Calcolo delle capacità e compatibilità per ogni richiesta
    num_compatibili_list = []
    cap_residua_list = []
    cap_occupata_domani_list = []
    
    for i in sorted_requests:
        req = requests[i]
        d = req['arrival_day']
        size_req = req['size']
        current_order = df_requests.loc[df_requests['i'] == i, 'order'].iloc[0]

        
        # Numero di locker compatibili
        num_compatibili = sum(phi_dict[(i, j)] for j in J) / len(J)
        
        cap_residua = 0
        cap_occupata_domani = 0
        
        for j in J:
            if phi_dict[(i, j)] == 1:
                for k in K:
                    if k >= size_req:
                        # Box liberi oggi + box che si liberano domani
                        available_today = C[(j, k)] - occupancy[(j, k, d)]
                        freeing_tomorrow = freeing[(j, k, d+1)]
                        cap_residua += available_today + freeing_tomorrow
                
                # Calcolo della capacità sicuramente occupata domani basato sulle richieste
                # assegnate fino al momento in cui arriva la richiesta i
                for prev_i, (prev_j, prev_k) in assigned_until_now.get(d, {}).items():
                    prev_order = df_requests.loc[df_requests['i'] == prev_i, 'order'].iloc[0]
                    prev_size = requests[prev_i]['size']
                    if prev_order < current_order and prev_j == j and prev_size >= size_req:
                        # Questa richiesta è stata assegnata prima dell'arrivo della richiesta corrente
                        # e al locker j compatibile con la richiesta corrente
                        # e con dimensione >= alla dimensione richiesta
                        cap_occupata_domani += 1
        
        num_compatibili_list.append(num_compatibili)
        cap_residua_list.append(cap_residua)
        cap_occupata_domani_list.append(cap_occupata_domani)
    
    df_requests['num_compatible'] = df_requests['i'].apply(lambda i: num_compatibili_list[sorted_requests.index(i)])
    df_requests['residual_cap'] = df_requests['i'].apply(lambda i: cap_residua_list[sorted_requests.index(i)])
    df_requests['occupied_cap_tom'] = df_requests['i'].apply(lambda i: cap_occupata_domani_list[sorted_requests.index(i)])
    
    
    
    # Calcolo POLICY per ogni richiesta
    # Il centro del quadrato è fisso (50, 50)
    center_coords = (50, 50)
    policy_dict = {}
    
    for i in data["I"]:
        arrival_day = requests[i]['arrival_day']
        size_req = int(requests[i]['size'])  # compartimento richiesto
        assigned = assignment[i]
        free_capacities = {}
        # Calcolo della capacità libera per ogni locker compatibile per il compartimento richiesto
        for j in J:
            for k in K:
                if phi_dict[(i, j)] == 1 and k >= size_req:
                    free_capacities[j] = C[(j, k)] - occupancy[(j, k, arrival_day)]
                
        policies = []
        # Se la richiesta non è assegnata, aggiungiamo direttamente la policy P6
        if assigned is None:
            policies.append("P6")
        else:
            if free_capacities:
                if len(free_capacities) == 1:
                    policies.append("P7")
                else:
                    # Determiniamo il locker con capacità libera massima e minima
                    j_max = max(free_capacities, key=lambda j: free_capacities[j])
                    j_min = min(free_capacities, key=lambda j: free_capacities[j])
                    # Determiniamo il locker più centrale e quello più periferico basandosi sulla distanza dal centro (50,50)
                    j_central = min(free_capacities, key=lambda j: np.sqrt((lockers[j]['x'] - center_coords[0])**2 + (lockers[j]['y'] - center_coords[1])**2))
                    j_peripheral = max(free_capacities, key=lambda j: np.sqrt((lockers[j]['x'] - center_coords[0])**2 + (lockers[j]['y'] - center_coords[1])**2))
                    
                    # Verifichiamo ogni condizione in modo indipendente e aggiungiamo la policy corrispondente se verificata
                    if assigned == j_max:
                        policies.append("P1")
                    if assigned == j_min:
                        policies.append("P2")
                    if assigned == j_central:
                        policies.append("P3")
                    if assigned == j_peripheral:
                        policies.append("P4")
                    # Se nessuna delle condizioni è soddisfatta, aggiungiamo P5
                    if not policies:
                        policies.append("P5")
            else:
                policies.append("P5")
        
        # Salviamo la lista delle policy (convertita in stringa separata da virgole) per la richiesta
        policy_dict[i] = ", ".join(policies)

    df_requests['policy'] = df_requests['i'].map(policy_dict)
    df_requests['status'] = df_requests['policy'].apply(lambda x: 1 if 'P6' not in x else 0)
    
    df_requests['arrival_day'] = df_requests['arrival_day'] / n_days  

    # Selezioniamo le colonne richieste (inclusa la policy)
    result_df = df_requests[['arrival_day', 'type', 'size', 'pickup_time',
                              'num_compatible', 'residual_cap',
                              'expected_future', 'expected_future_same_size', 'expected_future_type','occupied_cap_tom',
                              'policy', 'status']]
    
    result_df['request_id'] = result_df.index
    result_df['assigned_locker'] = result_df.index.map(lambda i: assignment[i] if i in assignment else None)
    #result_df['assigned_compartment'] = result_df.index.map(lambda i: assignment[i][1] if i in assignment else None)
    
    return result_df

def extract_ml_features(request, current_requests, occupancy, data):
    """
    Estrae le feature per il modello ML usando la stessa logica di extract_solution
    
    Parameters:
    - request: richiesta corrente
    - current_requests: lista delle richieste già processate nel giorno corrente
    - occupancy: dizionario che traccia l'occupazione dei compartimenti
    - data: dizionario con i dati del problema
    
    Returns:
    - Dictionary con le feature estratte
    """

    # Estrazione dati necessari
    J = data["J"]
    K = data["K"]
    C = data["C"] 
    phi_dict = data["phi_dict"]
    lockers = data["lockers"]
    
    # Feature di base della richiesta
    features = {
        'arrival_day': request['arrival_day'] / len(data["T"]),
        'type': 1 if request['type'] == 'premium' else 0,
        'size': request['size'],
        'pickup_time': request['pickup_time'],  #.get('pickup_time', 3),  # Default value as pickup time is unknown during assignment
        'order': request.get('order', 0)
    }
    
    # Numero di locker compatibili
    features['num_compatible'] = sum(phi_dict.get((request['request_id'], j), 0) for j in J) / len(J)
    
    # Calcolo capacità residua e occupata
    cap_residua = 0
    cap_occupata_domani = 0
    
    for j in J:
        if phi_dict.get((request['request_id'], j), 0) == 1:
            for k in K:
                if k >= request['size']:
                    # Box liberi domani (day + 1)
                    available_tomorrow = C[(j, k)] - occupancy.get((j, k, request['arrival_day'] + 1), 0)
                    cap_residua += available_tomorrow
            
            # Capacità occupata dalle richieste precedenti
            prev_requests = [r for r in current_requests if r.get('order', 0) < request.get('order', 0)]
            for prev_req in prev_requests:
                if phi_dict.get((prev_req['request_id'], j), 0) == 1:
                    if prev_req['size'] >= request['size']:
                        cap_occupata_domani += 1
    
    features['residual_cap'] = cap_residua
    features['occupied_cap_tom'] = cap_occupata_domani
    
    # Expected future requests - FISSATO: calcola su base giornaliera
    total_requests_per_day = 100  # assumiamo 100 richieste al giorno come nel dataset originale
    day_requests_remaining = total_requests_per_day - sum(1 for r in current_requests 
                                                       if r['arrival_day'] == request['arrival_day'] and 
                                                       r.get('order', 0) <= request.get('order', 0))
    features['expected_future'] = day_requests_remaining
    
    # Probabilità per size e type
    size_probs = {1: 0.5, 2: 0.3, 3: 0.2}
    type_probs = {'premium': 0.7, 'standard': 0.3}
    
    # FISSATO: Calcolo di expected_future_same_size e expected_future_type su base giornaliera
    # Conta le richieste già processate oggi dello stesso size/type
    same_size_today = sum(1 for r in current_requests 
                        if r['arrival_day'] == request['arrival_day'] and 
                        r['size'] == request['size'])
                        
    same_type_today = sum(1 for r in current_requests 
                        if r['arrival_day'] == request['arrival_day'] and 
                        r['type'] == request['type'])
    
    # Calcola le richieste attese per il resto della giornata
    expected_same_size = total_requests_per_day * size_probs[request['size']] - same_size_today
    expected_same_type = total_requests_per_day * type_probs[request['type']] - same_type_today
    
    features['expected_future_same_size'] = max(0, expected_same_size)
    features['expected_future_type'] = max(0, expected_same_type)

    # Feature aggiuntive sulla posizione dei locker
    center_coords = (50, 50)
    compatible_lockers = [j for j in J if phi_dict.get((request['request_id'], j), 0) == 1]
    
    if compatible_lockers:
        # Distanza dal centro per i locker compatibili
        distances = [np.sqrt((lockers[j]['x'] - center_coords[0])**2 + 
                           (lockers[j]['y'] - center_coords[1])**2) 
                    for j in compatible_lockers]
        
        features['min_distance_center'] = min(distances) if distances else -1
        features['max_distance_center'] = max(distances) if distances else -1
        features['avg_distance_center'] = sum(distances)/len(distances) if distances else -1
    else:
        features['min_distance_center'] = -1
        features['max_distance_center'] = -1
        features['avg_distance_center'] = -1
    
    #print(f"[DEBUG] Features for request {request['request_id']}: {features}")
    return features

def prepare_ml_input(request, current_state, data, mode = 'standard'):
    """
    Prepara l'input per il modello ML in formato numpy array
    """
    features = extract_ml_features(request, current_state['current_requests'], 
                                 current_state['temp_occupancy'], data)
    
    # Definizione dell'ordine delle feature per il modello
    if mode == 'stress':
        feature_order = [
            'type', 'size', 'pickup_time',
            'num_compatible', 'residual_cap',
            'expected_future', 'expected_future_same_size',
            'expected_future_type', 'occupied_cap_tom'
        ]
    else:
        feature_order = [
            'type', 'size',
            'num_compatible', 'residual_cap',
            'expected_future', 'expected_future_same_size', 
            'expected_future_type', 'occupied_cap_tom'
        ]
    
    #return [features[f] for f in feature_order]
    df = pd.DataFrame([features])[feature_order]
    return df

def check_capacity_constraints(request, data, current_state):
    """
    Verifies capacity constraints for request acceptance
    """
    J = data["J"]
    K = data["K"]
    C = data["C"]
    phi_dict = data["phi_dict"]
    arrival_day = request['arrival_day']
    next_day = arrival_day + 1
    size_req = request['size']
    request_id = request.get('request_id', request.get('i', 0))
    
    #print(f"\n[DEBUG] Checking capacity for Request {request_id} (Size {size_req})")
    #print(f"[DEBUG] Arrival day: {arrival_day}, Next day: {next_day}")
    
    # Reset temp_occupancy when moving to a new day
    if 'last_processed_day' not in current_state or current_state['last_processed_day'] != arrival_day:
        #print(f"[DEBUG] New day detected, resetting temp_occupancy")
        current_state['temp_occupancy'] = {}
        current_state['last_processed_day'] = arrival_day
        
        # Initialize temp_occupancy with real occupancy for next day
        for j in J:
            for k in K:
                current_state['temp_occupancy'][(j, k, next_day)] = current_state['occupancy'].get((j, k, next_day), 0)

    # Debug current occupancy state
    #print("\n[DEBUG] Current capacity states for next day:")
    #print("Format: Locker | Size | Temp occupancy/Capacity")
    #print("-" * 50)
    
    compatible_lockers = [j for j in J if phi_dict.get((request_id, j), 0) == 1]
    
    if not compatible_lockers:
        #print(f"[DEBUG] Request {request_id}: No compatible lockers found")
        return False
    
    #print(f"\n[DEBUG] Compatible lockers found: {compatible_lockers}")
    
    # Check for available capacity in compatible lockers
    for j in compatible_lockers:
        for k in sorted(K):
            if k >= size_req:
                temp_occ = current_state['temp_occupancy'].get((j, k, next_day), 0)
                total_cap = C.get((j, k), 0)
                #print(f"Locker {j} | Size {k} | {temp_occ}/{total_cap}")
                
                if temp_occ < total_cap:
                    #print(f"[DEBUG] Found available space in locker {j}, size {k}")
                    return True
    
    #print(f"[DEBUG] Request {request_id}: No capacity available in any compatible locker")
    return False

def predict_optimal_locker(request, data, current_state, model, mlb, mode = 'multilabel'):
    """
    Predice il locker ottimale basandosi sulle probabilità delle policy,
    verificando i vincoli di capacità per garantire che ci sia spazio per la richiesta
    """
    J = data["J"]
    K = data["K"]
    C = data["C"]
    phi_dict = data["phi_dict"]
    lockers = data["lockers"]
    size_req = request['size']
    arrival_day = request['arrival_day']
    next_day = arrival_day + 1
    
    #print(f"\n[DEBUG] Processing request {request['request_id']} (Day {arrival_day}, Size {size_req}, Type {request['type']})")
    
    # Prima verifica i vincoli globali di capacità
    if not check_capacity_constraints(request, data, current_state):
        #print(f"[DEBUG] Request {request['request_id']} rejected: Failed capacity constraints check")
        return None, 'capacity'
    
    # Calcolo capacità libere per locker compatibili
    feasible_lockers = {}
    
    for j in J:
        if phi_dict.get((request['request_id'], j), 0) == 1:
            # Verifica che il locker abbia almeno un compartimento disponibile per il giorno di assegnamento
            available_capacity = 0
            
            for k in K:
                if k >= size_req:
                    # Controlliamo solo per il giorno di assegnamento (day+1)
                    current_occupancy = current_state['temp_occupancy'].get((j, k, next_day), 0)
                    total_capacity = C.get((j, k), 0)
                    
                    if current_occupancy < total_capacity:
                        available_slots = total_capacity - current_occupancy
                        available_capacity += available_slots
            
            if available_capacity > 0:
                feasible_lockers[j] = available_capacity
                #print(f"[DEBUG] Locker {j} is feasible with {available_capacity} available slots")
    
    # Se non ci sono locker fattibili, rifiuta la richiesta
    if not feasible_lockers:
        #print(f"[DEBUG] Request {request['request_id']} rejected: No feasible lockers")
        return None, 'capacity'

    # Calcola metriche per ogni locker fattibile
    locker_metrics = {}
    center_coords = (50, 50)
    
    for j in feasible_lockers:
        distance = np.sqrt((lockers[j]['x'] - center_coords[0])**2 + 
                         (lockers[j]['y'] - center_coords[1])**2)
        locker_metrics[j] = {
            'capacity': feasible_lockers[j],
            'distance': distance
        }

    #X = prepare_ml_input(request, current_state, data)
    #X = np.array(X).reshape(1, -1)  # Reshape per il modello

    if mode == 'binary':
        X = prepare_ml_input(request, current_state, data)
        prediction = model.predict(X)[0]

        #print(f"[DEBUG] Binary prediction: {'accept' if prediction == 1 else 'reject'}")
        
        if prediction == 0:  # Reject
            #print(f"[DEBUG] Request {request['request_id']} rejected by binary model")
            return None, 'ml_reject'
        else:  # Accept and assign to nearest feasible locker
            # Calculate distances to all feasible lockers
            #locker_distances = {}
            #for j in feasible_lockers:
            #    distance = np.sqrt((lockers[j]['x'] - request['x'])**2 + 
            #                     (lockers[j]['y'] - request['y'])**2)
            #    locker_distances[j] = distance
            #
            ## Choose nearest feasible locker
            #nearest_locker = min(locker_distances.items(), key=lambda x: x[1])[0]
            locker_choice = min(feasible_lockers.items(), key=lambda x: x[1])[0]
            #print(f"[DEBUG] Request accepted: assigned to nearest locker {nearest_locker}")
            return locker_choice, 'accepted'
        
    elif mode == 'multilabel': # MULTILABEL
        X = prepare_ml_input(request, current_state, data)
        #X = np.array(X).reshape(1, -1)  # Reshape per il modello
        policy_probs = model.predict_proba(X).toarray()[0]  # Ottieni le probabilità per ogni policy
        
        # Converti in dizionario policy -> probabilità
        policy_dict = {}
        for policy, prob_array in zip(mlb.classes_, policy_probs):
            # Prendi il valore massimo di probabilità per ogni policy
            policy_dict[policy] = np.max(prob_array)
        
        #print(f"[DEBUG] Policy probabilities: {policy_dict}")
        
        # Scegli la policy con la probabilità più alta
        if 'P6' in policy_dict and policy_dict['P6'] > 0.5:# and all(p<0.5 for p in policy_dict.values()):
            # Rifiuta la richiesta
            #print(f"[DEBUG] Request {request['request_id']} rejected: Policy P6 selected")
            return None, 'ml_reject'
        else:
            # Escludi P6 per trovare la policy migliore tra le altre
            best_policies = {p: prob for p, prob in policy_dict.items() if p != 'P6' and prob > 0.5}
            if not best_policies:
                locker_distances = {}
                for j in feasible_lockers:
                    distance = np.sqrt((lockers[j]['x'] - request['x'])**2 + 
                                    (lockers[j]['y'] - request['y'])**2)
                    locker_distances[j] = distance
                locker_choice = min(locker_distances.items(), key=lambda x: x[1])[0]
                #print(f"[DEBUG] Request accepted: assigned to nearest locker {locker_choice}")
                return locker_choice, 'accepted'
            
            best_policy = max(best_policies, key=best_policies.get)
            #print(f"[DEBUG] Selected policy: {best_policy}")
            
            # Applica la policy selezionata sui locker fattibili
            if best_policy == 'P1':  # Massima capacità
                locker_choice = max(feasible_lockers.items(), key=lambda x: x[1])[0]
            elif best_policy == 'P2':  # Minima capacità
                locker_choice = min(feasible_lockers.items(), key=lambda x: x[1])[0]
            elif best_policy == 'P3':  # Più centrale
                locker_choice = min(locker_metrics.items(), key=lambda x: x[1]['distance'])[0]
            elif best_policy == 'P4':  # Più periferico
                locker_choice = max(locker_metrics.items(), key=lambda x: x[1]['distance'])[0]
            elif best_policy == 'P5':  # Bilanciamento carico
                # Trova il locker con il rapporto carico/capacità più basso
                locker_distances = {}
                for j in feasible_lockers:
                    distance = np.sqrt((lockers[j]['x'] - request['x'])**2 + 
                                    (lockers[j]['y'] - request['y'])**2)
                    locker_distances[j] = distance
                locker_choice = min(locker_distances.items(), key=lambda x: x[1])[0]
                #print(f"[DEBUG] Request accepted: assigned to nearest locker {locker_choice}")
                return locker_choice, 'accepted'
            

            #print(f"[DEBUG] Assigned locker {locker_choice} based on policy {best_policy}")
            return locker_choice, 'accepted'

    elif mode == 'ml_stress':
        X = pd.DataFrame(prepare_ml_input(request, current_state, data, mode = 'stress'))
        predictions = []
        prob = [0.4, 0.3, 0.3]
        
        for i in range(1, 4):
            
            request_copy = X.copy()
            request_copy['pickup_time'] = i
            
            
            predictions.append(model.predict_proba(request_copy)[0][1])

        w_avg = np.dot(prob, predictions)
        if w_avg > 0.5:
            locker_choice = max(feasible_lockers.items(), key=lambda x: x[1])[0]  #max capacity
            #locker_distances = {}
            #for j in feasible_lockers:
            #    distance = np.sqrt((lockers[j]['x'] - request['x'])**2 + 
            #                    (lockers[j]['y'] - request['y'])**2)
            #    locker_distances[j] = distance
            #locker_choice = min(locker_distances.items(), key=lambda x: x[1])[0]
            return locker_choice, 'accepted'
        else:
            return None, 'ml_reject'


    elif mode == 'greedy':   
        locker_choice = max(feasible_lockers.items(), key=lambda x: x[1])[0]
        #print(f"[DEBUG] Selected locker {locker_choice} using default policy (max capacity)")
        return locker_choice, 'accepted'

def process_end_of_day_assignments(current_state, day, data, results):
    """
    Process end-of-day compartment assignments with support for extended days.
    
    In this phase, we know which packages will be picked up for each accepted request,
    but for compartment assignment we pretend not to know the pickup_time
    and only guarantee space for the package the day after arrival.
    """
    #print(f"\n[DEBUG] Processing end-of-day assignments for day {day}")
    
    if day not in current_state['pending_assignments']:
        #print(f"[DEBUG] No pending assignments for day {day}")
        return
    
    pending = current_state['pending_assignments'][day]
    
    # Debug: show pending requests
    #print(f"[DEBUG] Pending assignments for day {day}: {len(pending)} requests")
    #for i, (req, locker) in enumerate(pending):
        #print(f"[DEBUG] {i+1}: Request ID {req['request_id']}, Size {req['size']}, Type {req['type']}, Locker {locker}")
    
    # Sort by type (premium first) and then by size (smaller first)
    #pending.sort(key=lambda x: (0 if x[0]['type'] == 'premium' else 1, x[0]['size']))
    
    #print(f"[DEBUG] Sorted pending assignments:")
    #for i, (req, locker) in enumerate(pending):
        #print(f"[DEBUG] {i+1}: Request ID {req['request_id']}, Size {req['size']}, Type {req['type']}, Locker {locker}")
    
    J = data["J"]
    K = data["K"]
    
    for request, assigned_locker in pending:
        size_req = request['size']
        pickup_time = modify_pickup(data,assigned_locker)  # This is known during simulation but not to the system
        pickup_day = day + pickup_time
        
        # Ensure pickup_day doesn't exceed max_day (13)
        #pickup_day = min(pickup_day, 13)
        
        #print(f"\n[DEBUG] Assigning compartment for request {request['request_id']} to locker {assigned_locker}")
        #print(f"[DEBUG] Request size: {size_req}, Pickup time: {pickup_time}, Pickup day: {pickup_day}")
        
        # Look for the smallest compatible compartment available
        assigned_k = None
        for k in sorted(K):  # Sort K to look for smaller compartments first
            if k >= size_req:  # The compartment must be at least the required size
                # Check if there's space for the assignment day
                current_occupancy = current_state['occupancy'].get((assigned_locker, k, day + 1), 0)
                total_capacity = data['C'].get((assigned_locker, k), 0)
                
                #print(f"[DEBUG] Checking compartment size {k}: occupancy {current_occupancy}/{total_capacity}")
                
                if current_occupancy < total_capacity:
                    # Assign this compartment
                    assigned_k = k
                    #print(f"[DEBUG] Assigned compartment size {k}")
                    
                    # Update occupancy for all days the package will be in the locker
                    for d in range(day + 1, pickup_day + 1):
                        current_occ = current_state['occupancy'].get((assigned_locker, k, d), 0)
                        current_state['occupancy'][(assigned_locker, k, d)] = current_occ + 1
                        #print(f"[DEBUG] Updated occupancy for day {d}: now {current_occ + 1}/{total_capacity}")
                    break
        
        # Update results with compartment assignment
        assigned = False
        for result in results:
            if result['request_id'] == request['request_id']:
                result['compartment_size'] = assigned_k
                
                # If it wasn't possible to assign a compartment, change status to rejected
                if assigned_k is None:
                    result['status'] = 'rejected'
                    result['assigned_locker'] = None
                    #print(f"[ERROR] Request {request['request_id']} was accepted but no compartment found at assignment time!")
                #else:
                    #print(f"[DEBUG] Request {request['request_id']} successfully assigned to compartment size {assigned_k}")
                
                assigned = True
                break
        
        if not assigned:
            print(f"[ERROR] Could not find result record for request {request['request_id']}")
    
    # Clear processed assignments
    del current_state['pending_assignments'][day]

def ML_policy(data, model=None, mlb=None, mode = 'multilabel'):
    """
    Two-phase allocation policy implementing the problem structure
    Extended to track occupancy for 3 additional days after last request
    """
    #print("\n[DEBUG] Starting ML_policy simulation with extension to day 13...")
    
    J = data["J"]
    K = data["K"]
    max_day = 13  
    
    # Initialize system state
    #current_state = {
    #    'pending_assignments': {},  # Requests waiting for compartment assignment
    #    'current_requests': []  # Requests processed so far
    #    'occupancy': {(j, k, d): 0 for j in J for k in K for d in range(max_day+1)},  # Compartment occupancy
    #}
    
    current_state = {
        'occupancy': {},  # (j, k, t) -> occupancy count
        'temp_occupancy': {},  # Temporary occupancy for day's decisions
        'pending_assignments': {} , # day -> list of (request, locker) awaiting end-of-day assignment
        'current_requests': [],  # List of requests processed so far
        'last_processed_day': None
    
    }
    C = data["C"]
    results = []  # List to save request results
    current_day = -1  # Current day initialized to -1
    
    # Sort requests by arrival day and order number
    sorted_requests = sorted(data['I'], 
                           key=lambda x: (data['requests'][x]['arrival_day'], 
                                        data['requests'][x].get('order', x)))
    
    #print(f"[DEBUG] Total requests to process: {len(sorted_requests)}")
    
    for i in sorted_requests:
        # Create a copy of the request to avoid unwanted modifications
        request = data['requests'][i].copy()
        request['i'] = i  # Request ID for compatibility with existing code
        request['request_id'] = i  # Make sure request_id is present
        day = request['arrival_day']
        size = request['size']
        # Process end-of-day assignments for previous day
        if day > current_day and current_day >= 0:
            #print(f"\n[DEBUG] Day change detected: {current_day} -> {day}")
            process_end_of_day_assignments(current_state, current_day, data, results)
            #visualize_occupancy(current_state, data, current_day + 1)  # Visualize state after assignment
        
        current_day = day
        
        # Decide whether to accept or reject the request
        #print(f"\n[DEBUG] Day {day}: Processing request {i}")
        assigned_locker, decision_reason = predict_optimal_locker(request, data, current_state, model, mlb, mode)

        # If the request is accepted, add to the list of requests waiting for compartment assignment
        #if assigned_locker is not None:
        #    if day not in current_state['pending_assignments']:
        #        current_state['pending_assignments'][day] = []
        #    current_state['pending_assignments'][day].append((request, assigned_locker))
        #    #key = (assigned_locker, size, day + 1)
        #    #current_state['temp_occupancy'][key] = current_state['temp_occupancy'].get(key, 0) + 1
        #    current_state['temp_occupancy'][(assigned_locker, k, next_day)] = current_state['temp_occupancy'].get((assigned_locker, k, next_day), 0) + 1
        #    status = decision_reason
        #    print(f"[DEBUG] Request {i} accepted and assigned to locker {assigned_locker}")
        #else:
        #    status = decision_reason
        #    print(f"[DEBUG] Request {i} rejected")
            # Double-check capacity for the assigned locker before final acceptance
            # Inside ML_policy function, replace the if assigned_locker is not None block with:

        if assigned_locker is not None:
            # Check capacity and update temp_occupancy if space is available
            capacity_available = False
            size = request['size']
            
            for k in sorted(K):
                if k >= size:
                    temp_occ = current_state['temp_occupancy'].get((assigned_locker, k, current_day+1), 0)
                    total_capacity = C.get((assigned_locker, k), 0)
                    
                    if temp_occ < total_capacity:
                        capacity_available = True
                        current_state['temp_occupancy'][(assigned_locker, k, current_day+1)] = temp_occ + 1
                        
                        # Add to pending assignments
                        if current_day not in current_state['pending_assignments']:
                            current_state['pending_assignments'][current_day] = []
                        current_state['pending_assignments'][current_day].append((request, assigned_locker))
                        status = 'accepted'
                        break
            
            if not capacity_available:
                #print(f"[DEBUG] Request {i}: No capacity available in assigned locker {assigned_locker}")
                status = 'capacity'
                assigned_locker = None
        else:
            status = decision_reason
        
        # Save results
        results.append({
            'request_id': i,
            'arrival_day': day,
            'type': request['type'],
            'size': request['size'],
            'assigned_locker': assigned_locker,
            'compartment_size': None,  # Will be updated during compartment assignment
            'status': status,
            'pickup_time': request['pickup_time']  # This is known in the simulation, but not during assignment
        })
        
        # Add the request to those processed
        current_state['current_requests'].append(request)
    
    # Process last day's assignments
    if current_day >= 0:
        #print(f"\n[DEBUG] Processing final end-of-day assignments for day {current_day}")
        process_end_of_day_assignments(current_state, current_day, data, results)
    
    # Visualize occupancy for the extended days (days 11, 12, 13)
    #print("\n[DEBUG] Visualizing occupancy for extended days:")
    #for extended_day in range(current_day + 2, max_day + 1):
    #    visualize_occupancy(current_state, data, extended_day)
    
    # Final statistics
    total_requests = len(results)
    accepted_requests = sum(1 for r in results if r['status'] == 'accepted')
    premium_requests = sum(1 for r in results if r['type'] == 'premium')
    premium_accepted = sum(1 for r in results if r['type'] == 'premium' and r['status'] == 'accepted')

    # statistiche motivi rifiutate 
    rejected_requests_ml = sum(1 for r in results if r['status'] == 'ml_reject')
    rejected_requests_capacity = sum(1 for r in results if r['status'] == 'capacity')

    rejected_requests_ml_premium = sum(1 for r in results if r['status'] == 'ml_reject' and r['type'] == 'premium')
    rejected_requests_capacity_premium = sum(1 for r in results if r['status'] == 'capacity' and r['type'] == 'premium')
    rejected_requests_ml_standard = sum(1 for r in results if r['status'] == 'ml_reject' and r['type'] == 'standard')
    rejected_requests_capacity_standard = sum(1 for r in results if r['status'] == 'capacity' and r['type'] == 'standard')

    
    print("\n[DEBUG] Final statistics:")
    print(f"[DEBUG] Total requests: {total_requests}")
    print(f"[DEBUG] Accepted requests: {accepted_requests} ({accepted_requests/total_requests*100:.2f}%)")
    print(f"[DEBUG] Premium requests: {premium_requests}")
    print(f"[DEBUG] Premium accepted: {premium_accepted} ({premium_accepted/premium_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests: {total_requests - accepted_requests} ({(total_requests - accepted_requests)/total_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (ML): {rejected_requests_ml} ({rejected_requests_ml/total_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (Capacity): {rejected_requests_capacity} ({rejected_requests_capacity/total_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (ML Premium): {rejected_requests_ml_premium} ({rejected_requests_ml_premium/premium_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (Capacity Premium): {rejected_requests_capacity_premium} ({rejected_requests_capacity_premium/premium_requests*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (ML Standard): {rejected_requests_ml_standard} ({rejected_requests_ml_standard/(total_requests - premium_requests)*100:.2f}%)")
    print(f"[DEBUG] Rejected requests (Capacity Standard): {rejected_requests_capacity_standard} ({rejected_requests_capacity_standard/(total_requests - premium_requests)*100:.2f}%)")


    
    # Additional occupancy statistics for extended days
    #print("\n[DEBUG] Extended days occupancy statistics:")
    #for day in range(11, max_day + 1):
    #    total_occupied = sum(current_state['occupancy'].get((j, k, day), 0) 
    #                        for j in J for k in K)
    #    total_capacity = sum(data['C'][(j, k)] for j in J for k in K)
    #    #print(f"[DEBUG] Day {day}: {total_occupied}/{total_capacity} compartments occupied " +
    #          f"({total_occupied/total_capacity*100:.2f}%)")
    
    return pd.DataFrame(results)

def oracle_optimization():
    """
    Function to create 6 datasets (2x4 lockers, 2x6 lockers, 2x8 lockers)
    and run the oracle optimization for each dataset. The results are saved in Excel files.
    """
    datasets = {}
    results = {}
    locker_configs = [(4, 2), (6, 2), (8, 2)]  # (n_lockers, n_datasets)

    counter = 1
    for n_lockers, n_datasets in locker_configs:
        for _ in range(n_datasets):  
            data = prepare_data(n_days=10, n_requests_per_day=100, n_lockers=n_lockers)
            datasets[counter] = data
            print(f"---- Dataset {counter} created with {n_lockers} lockers ----")
            print(f"\n--- Optimization dataset {counter} ---")
            m = oracle(data)
            data = extract_solution(data)
            results[counter] = data
            counter += 1

    # Save df_requests to Excel
    with pd.ExcelWriter('instances.xlsx') as writer:
        for counter, data in datasets.items():
            sheet_name = f"Instance_{counter}"
            data['df_requests'].to_excel(writer, sheet_name=sheet_name, index=False)
    print("\nInstances saved in: instances.xlsx")

    # Save df_lockers to Excel
    with pd.ExcelWriter('lockers.xlsx') as writer:
        for counter, data in datasets.items():
            sheet_name = f"Lockers_{counter}"
            data['df_lockers'].to_excel(writer, sheet_name=sheet_name, index=False)
    print("Lockers salvati nel file lockers.xlsx")

    # Save results to Excel
    with pd.ExcelWriter('results_oracle.xlsx') as writer:
        for counter, data in results.items():
            sheet_name = f"Instance_{counter}"
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Results saved in results_oracle.xlsx")

    # Sample requests from each dataset
    sampled_df = pd.DataFrame()
    for counter, data in results.items():
        sampled_requests = sample_requests_per_day(data, 50)
        sampled_df = pd.concat([sampled_df, sampled_requests], ignore_index=True)

    # Process the sampled data
    sampled_df['type'] = sampled_df['type'].apply(lambda x: 1 if x == 'premium' else 0)
    sampled_df['policy'] = sampled_df['policy'].apply(lambda x: x.split(', '))
    sampled_df['status'] = sampled_df['policy'].apply(lambda x: 1 if 'P6' not in x else 0)

    # Create complete dataset without P7
    complete_df = pd.concat(results.values(), ignore_index=True)
    complete_df = complete_df[~complete_df['policy'].str.contains('P7')]

    # Save sampled dataset to Excel
    sampled_df.to_excel('ml_dataset.xlsx', index=False)
    print("\nML-dataset saved to ml_dataset.xlsx")

    # Visualize policy distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    complete_df['policy'].value_counts().plot(kind='bar')
    plt.title('Policy Distribution - Complete Dataset')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sampled_df['policy'].value_counts().plot(kind='bar')
    plt.title('Policy Distribution - Sampled Dataset')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Complete dataset size: {len(complete_df)}")
    print(f"Sampled dataset size: {len(sampled_df)}")
    print("\nPolicy distributions in sampled data:")
    print(sampled_df['policy'].explode().value_counts(normalize=True))

    return complete_df, sampled_df

def ml_models(data, target='policy', mode = 'standard'):
    
    numerical_features = ['num_compatible', 'residual_cap', 'expected_future', 'expected_future_same_size', 'expected_future_type', 'occupied_cap_tom']
    if mode == 'stress':
        categorical_features = ['size', 'pickup_time']  # Include pickup_time for stress test
    else:
        categorical_features = ['size']
    others = ['type']

    preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features),
                ('other', 'passthrough', others)
            ], remainder='drop')
    
    if target == 'policy':
        # Multilabel Classification
        X = data.drop(columns=['request_id','policy', 'status'])
        y = data['policy']

        # Binarize multilabel target
        unique_classes = set(sum(y.tolist(), []))
        mlb = MultiLabelBinarizer(classes=list(unique_classes))
        y_encoded = mlb.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        base_rf = RandomForestClassifier(
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_estimators=100,
                    max_depth=None
                )
        
        #base_rf = GradientBoostingClassifier(
        #    class_weight='balanced_subsample',
        #    random_state=42,
        #    n_estimators=100)
#
        #base_rf = LogisticRegression()
        #base_rf = DecisionTreeClassifier()
        #base_rf = SVC(probability=True, random_state=42)
        #base_rf = MLPClassifier(random_state=42, max_iter=500)

        
        br_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', BinaryRelevance(classifier=base_rf))
        ])
        cc_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ClassifierChain(classifier=base_rf))
        ])
        
        br_pipeline.fit(X_train, y_train)
        cc_pipeline.fit(X_train, y_train)



        for model, name in [(br_pipeline, "Binary Relevance"), (cc_pipeline, "Classifier Chain")]:
            y_pred = model.predict(X_test)
            print("--" * 40)
            print(f"Classification Report {name}:")
            print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
            print("Hamming Loss:", hamming_loss(y_test, y_pred))

        if mode == 'standard':
            joblib.dump(br_pipeline, 'ml_models/br_model_p4.joblib')
            joblib.dump(cc_pipeline, 'ml_models/cc_model_p4.joblib')
            joblib.dump(mlb, 'ml_models/mlb_pipeline_p4.joblib')
            print("Saved multilabel classification models")
            return br_pipeline, cc_pipeline, mlb

    else:
        # Binary Classification
        print(data.columns)
        X = data.drop(columns=['request_id','policy', 'status'])
        y = data['status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        base_rf = RandomForestClassifier(random_state=42)
        #base_rf = GradientBoostingClassifier(random_state = 42)
        #base_rf = DecisionTreeClassifier(random_state=42)
        #base_rf = MLPClassifier(random_state=42, max_iter=500)
        #base_rf = SVC(probability=True, random_state=42)

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_rf)
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }

        #param_grid = {
        #    'classifier__max_depth': [None, 3, 5, 10],
        #    'classifier__min_samples_split': [2, 5],
        #    'classifier__criterion': ['gini', 'entropy']
        #}



        
        search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=5, verbose=0)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        print("Classification Report (Binary):")
        print(classification_report(y_test, y_pred))

        # Feature importance
        #rf = best_model.named_steps['classifier']
        #importances = rf.feature_importances_
        #sorted_idx = np.argsort(importances)[::-1]
        #plt.figure(figsize=(12,6))
        #plt.bar(np.array(X.columns)[sorted_idx], importances[sorted_idx])
        #plt.xticks(rotation=45, ha='right')
        #plt.title('Feature Importance')
        #plt.tight_layout()
        #plt.show()

        if mode == 'standard':
            joblib.dump(best_model, 'ml_models/binary_model_p4.joblib')
            print("Saved binary classification model")
            return best_model
        elif mode == 'stress':
            joblib.dump(best_model, 'ml_models/stress_model_p4.joblib')
            print("Saved stress test model")
            return best_model

def load_ml_models():
    """
    Loads all trained ML models from disk
    
    Returns:
        tuple: (binary_model, br_model, cc_model, mlb, stress_model)
    """
    try:
        binary_model = joblib.load('ml_models/binary_model.joblib')
        br_model = joblib.load('ml_models/br_model.joblib')
        cc_model = joblib.load('ml_models/cc_model.joblib')
        mlb = joblib.load('ml_models/mlb_pipeline.joblib')
        stress_model = joblib.load('ml_models/stress_model.joblib')
        
        print("Successfully loaded all models")
        return binary_model, br_model, cc_model, stress_model, mlb
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None

def modify_pickup_to_4_days(df_requests):
    """
    Modifica il sistema di pick-up da 3 a 4 giorni secondo le regole specificate:
    - 2/3 dei pick-up 3 passano a pick-up 4
    - 1/3 dei pick-up 3 + 1/3 dei pick-up 2 passano a pick-up 3
    - 1/3 dei pick-up 2 rimane pick-up 2
    - Tutti i pick-up 1 rimangono pick-up 1
    """
    df_modified = df_requests.copy()
    
    # Seleziona le righe per ogni tipo di pick-up
    pickup_1_mask = df_modified['pickup_time'] == 1
    pickup_2_mask = df_modified['pickup_time'] == 2
    pickup_3_mask = df_modified['pickup_time'] == 3
    
    # Conta i numeri per debugging
    n_pickup_1 = pickup_1_mask.sum()
    n_pickup_2 = pickup_2_mask.sum()
    n_pickup_3 = pickup_3_mask.sum()
    
    print(f"Distribuzione originale:")
    print(f"Pick-up 1: {n_pickup_1}")
    print(f"Pick-up 2: {n_pickup_2}")
    print(f"Pick-up 3: {n_pickup_3}")
    
    # Applica le regole di transizione
    
    # 1. Pick-up 1 rimangono tutti pick-up 1 (nessuna modifica necessaria)
    
    # 2. Pick-up 2: 1/3 rimane 2, 1/3 passa a 3
    if n_pickup_2 > 0:
        pickup_2_indices = df_modified[pickup_2_mask].index.tolist()
        np.random.shuffle(pickup_2_indices)
        
        # Calcola i numeri per la divisione
        third_pickup_2 = len(pickup_2_indices) // 3
        remainder_2 = len(pickup_2_indices) % 3
        
        # 1/3 rimane pick-up 2 (primi third_pickup_2 elementi)
        stay_2_indices = pickup_2_indices[:third_pickup_2]
        
        # 1/3 passa a pick-up 3 (successivi third_pickup_2 elementi)
        to_3_from_2_indices = pickup_2_indices[third_pickup_2:2*third_pickup_2]
        
        # Il resto rimane pick-up 2 (gestisce il remainder)
        remainder_2_indices = pickup_2_indices[2*third_pickup_2:]
        
        # Applica le modifiche
        df_modified.loc[to_3_from_2_indices, 'pickup_time'] = 3
        
        print(f"Pick-up 2 -> Pick-up 2: {len(stay_2_indices) + len(remainder_2_indices)}")
        print(f"Pick-up 2 -> Pick-up 3: {len(to_3_from_2_indices)}")
    
    # 3. Pick-up 3: 2/3 passa a 4, 1/3 rimane 3
    if n_pickup_3 > 0:
        pickup_3_indices = df_modified[pickup_3_mask].index.tolist()
        np.random.shuffle(pickup_3_indices)
        
        # Calcola i numeri per la divisione
        two_thirds_pickup_3 = (2 * len(pickup_3_indices)) // 3
        one_third_pickup_3 = len(pickup_3_indices) - two_thirds_pickup_3
        
        # 2/3 passa a pick-up 4
        to_4_indices = pickup_3_indices[:two_thirds_pickup_3]
        
        # 1/3 rimane pick-up 3
        stay_3_indices = pickup_3_indices[two_thirds_pickup_3:]
        
        # Applica le modifiche
        df_modified.loc[to_4_indices, 'pickup_time'] = 4
        
        print(f"Pick-up 3 -> Pick-up 3: {len(stay_3_indices)}")
        print(f"Pick-up 3 -> Pick-up 4: {len(to_4_indices)}")
    
    # Verifica la distribuzione finale
    print(f"\nDistribuzione finale:")
    for pickup_day in sorted(df_modified['pickup_time'].unique()):
        count = (df_modified['pickup_time'] == pickup_day).sum()
        print(f"Pick-up {pickup_day}: {count}")
    
    return df_modified

def calculate_distance(customer_pos, locker_pos):
    """Calcola la distanza euclidea tra cliente e locker"""
    return np.sqrt((customer_pos[0] - locker_pos[0])**2 + (customer_pos[1] - locker_pos[1])**2)

def oracle_new(data):
    """The modified constraints (5a-5f) impose that a parcel occupies the compartment to which it has been assigned for a duration that depends on both the original pickup time $\alpha_i$ and the distance $d_{ij}$ to the assigned locker. The pickup time is reduced when the assigned locker is sufficiently close to the customer: for $\alpha_i=2$, it reduces to 1 day if $d_{ij} \leq 10$; for $\alpha_i=3$, it reduces to 1 day if $d_{ij} \leq 10$ or to 2 days if $d_{ij} \leq 20$."""
    T = data["T"]
    I = data["I"]
    J = data["J"]
    K = data["K"]
    requests = data["requests"]
    lockers = data["lockers"]
    phi_dict = data["phi_dict"]
    C = data["C"]

    max_pickup = max(requests[i]['pickup_time'] for i in I)
    n_days = len(T)
    T_ext = set(range(n_days + max_pickup + 1))

    # Creazione del modello Gurobi
    model = gb.Model('ParcelLockerOracle')

    # Definizione delle variabili (definite come globali per essere usate in extract_solution)
    global x, y, z, u
    x = model.addVars(I, vtype=GRB.BINARY, name='x')
    y = model.addVars(I, J, vtype=GRB.BINARY, name='y')
    z = model.addVars(I, J, K, vtype=GRB.BINARY, name='z')
    u = model.addVars(I, J, K, T_ext, vtype=GRB.BINARY, name='u')

    # CONSTRAINTS
    # (1) Una richiesta viene assegnata ad uno e un solo locker se è accettata
    for i in I:
        model.addConstr(gb.quicksum(y[i, j] for j in J) == x[i])
    
    # (2) Solo i locker compatibili possono essere assegnati
    for i in I:
        for j in J:
            if phi_dict[(i, j)] == 0:
                model.addConstr(y[i, j] == 0)
    
    # (3) Per ciascuna richiesta, assegnazione del compartimento di tipo sufficiente
    for i in I:
        for j in J:
            model.addConstr(gb.quicksum(z[i, j, k] for k in K if k >= requests[i]['size']) == y[i, j])
    
    # (4) MODIFICA: Relazione tra z e u con riduzione pickup time basata su distanza
    for i in I:
        arrival_day = requests[i]['arrival_day']
        original_pickup_time = requests[i]['pickup_time']
        customer_pos = (requests[i]['x'], requests[i]['y'])  # Assumendo che le coordinate siano nel dizionario
        
        for j in J:
            locker_pos = (lockers[j]['x'], lockers[j]['y'])  # Assumendo che le coordinate siano nel dizionario
            distance = calculate_distance(customer_pos, locker_pos)
            
            # Calcolo del pickup time ridotto basato sulla distanza
            if original_pickup_time == 1:
                # Per pickup time 1 non cambia nulla
                effective_pickup_time = 1
            elif original_pickup_time == 2:
                # Si riduce a 1 se distanza <= 10
                if distance <= 10:
                    effective_pickup_time = 1
                else:
                    effective_pickup_time = 2
            elif original_pickup_time == 3:
                # Si riduce a 1 se distanza <= 10, a 2 se distanza <= 20
                if distance <= 10:
                    effective_pickup_time = 1
                elif distance <= 20:
                    effective_pickup_time = 2
                else:
                    effective_pickup_time = 3
            else:
                # Per pickup time > 3, mantieni la logica originale
                effective_pickup_time = original_pickup_time
            
            for k in K:
                for t in range(1, effective_pickup_time + 1):
                    t_index = arrival_day + t
                    model.addConstr(u[i, j, k, t_index] == z[i, j, k])
    
    # (5) Capacità dei locker: ad ogni istante t, il numero di compartimenti occupati non supera la capacità
    for t in T_ext:
        for j in J:
            for k in K:
                model.addConstr(gb.quicksum(u[i, j, k, t] for i in I) <= C[(j, k)])
    
    # OBJECTIVE FUNCTION: massimizzare la somma dei pesi delle richieste assegnate
    model.setObjective(gb.quicksum(requests[i]['weight'] * x[i] for i in I), GRB.MAXIMIZE)
    model.update()
    # Facoltativo: model.write(f'model_{n_lockers}_lockers.lp')
    model.setParam('OutputFlag', 0) 
    model.optimize()
    return(model)

def modify_pickup(data, assigned_locker):
    """Modifica il pickup time di una richiesta in base alla distanza dal locker assegnato"""
    data = data.copy()
    lockers = data['lockers']
    J = data['J']
    I = data['I']
    requests = data['df_requests']
    for i in I:
        original_pickup_time = requests.loc[i-1]['pickup_time']
        customer_pos = (requests.loc[i-1]['x'], requests.loc[i-1]['y'])  # Assumendo che le coordinate siano nel dizionario

        locker_pos = (lockers[assigned_locker]['x'], lockers[assigned_locker]['y'])  # Assumendo che le coordinate siano nel dizionario
        distance = calculate_distance(customer_pos, locker_pos)
        
        # Calcolo del pickup time ridotto basato sulla distanza
        if original_pickup_time == 1:
            # Per pickup time 1 non cambia nulla
            effective_pickup_time = 1
            return effective_pickup_time
        elif original_pickup_time == 2:
            # Si riduce a 1 se distanza <= 10
            if distance <= 10:
                effective_pickup_time = 1
                return effective_pickup_time
            else:
                effective_pickup_time = 2
                return effective_pickup_time
        elif original_pickup_time == 3:
            # Si riduce a 1 se distanza <= 10, a 2 se distanza <= 20
            if distance <= 10:
                effective_pickup_time = 1
                return effective_pickup_time
            elif distance <= 20:
                effective_pickup_time = 2
                return effective_pickup_time
            else:
                effective_pickup_time = 3
                return effective_pickup_time
        else:
            # Per pickup time > 3, mantieni la logica originale
            effective_pickup_time = original_pickup_time
            return effective_pickup_time



