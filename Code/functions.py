import pandas as pd
import numpy as np
import geopandas as gpd
import gurobipy as gp
from shapely import Point, LineString
from shapely import wkt
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


def Subset(elements, types):
    return elements[elements["type"].isin(types)]

def import_elements(file):
    
    raw_elements = pd.read_excel(file)

    raw_elements['coor'] = raw_elements['coor'].apply(wkt.loads)
    elements = gpd.GeoDataFrame(raw_elements, crs='epsg:4326', geometry='coor')
    elements["length"] = elements.length

    visualize_elements(elements)

    return elements

def visualize_elements(elements, connections=None):
    # Define a color map based on the values in 'terminal'
    unique_terminals = elements['terminal'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_terminals))  # Use a colormap with a number of colors equal to unique terminals

    # Create a dictionary mapping terminal values to colors
    color_map = {terminal: colors(i) for i, terminal in enumerate(unique_terminals)}

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot elements with colors based on 'terminal'
    elements['color'] = elements['terminal'].map(color_map)
    elements.plot(ax=ax, color=elements['color'], markersize=50, label='Elements')

    if(connections is not None):
        # Plot connections
        connections['color'] = connections['terminal'].map(color_map)
        connections.plot(ax=ax, color=connections['color'], linestyle='--', linewidth=.5, label='Connections')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Elements Plot')
    fig.legends = []
    plt.tight_layout()
    plt.show()



class Path:
    def __init__(self, elements):
        self.elements = elements

    def __eq__(self, other):
        if len(self.elements) != len(other.elements):
            return False
        return all(self.elements[i].name == other.elements[i].name for i in range(len(self.elements)))
    def __len__(self):
        return len(self.elements)

    def get_length(self):
        return sum(element.length for element in self.elements)

    def get_names(self):
        return ", ".join(e.name for e in self.elements)

    def get_elements_names(self):
        return [e.name for e in self.elements]
    def get_elements(self):
        return self.elements

class Node():
    """An element class to represent a node for A* Pathfinding"""

    def __init__(self, parent=None, element=None, g=1):
        self.parent = parent
        self.name = element["name"]
        self.position = element['coor']
        self.type = element['type']
        self.length = element['length']

        self.g = g
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.name == other.name

    def distance(self, other):
        return self.position.distance(other.position)

    def connectable(self, other, D):
        return self.name != other.name and self.position.distance(other["coor"]) < D #and self.type == other["type"]

def astar(elements, element_start, num_paths, D=100):
    # Create end and end start
    end_element = elements[elements['name']==element_start["terminal"]]
    end_node = Node(None, end_element.squeeze() )
    end_node.g = end_node.h = end_node.f = 0

    start_node = Node(None, element_start)
    start_node.g = 0 
    start_node.h = start_node.f = start_node.distance(end_node)

    initial_g = 2
    g_multiplier = 3

    paths = []
    g_weights = {}
    possible_connections = pd.concat([Subset(elements, ["line"]), end_element], ignore_index=True)
    for c in range(num_paths):
        # Initialize both open and closed list
        open_list = [] #You can use a PriorityQueue to avoid looping through the list to find the node with the smallest f-value but you can not check if an element is inside the PriorityQueue.
        closed_list = []

        # Add the start node
        open_list.append(start_node)
        # Loop until you explored all the elements (or you found the end)
        while len(open_list) > 0:
            # Get the current node: the node with the lowest f-value
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current node off from the open list. Add it to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the end node. We stop
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent

                p = Path(path[::-1]) #reverse path
                if(p in paths):
                    continue
                else:
                    paths.append(p) # append the reversed path
                    # Update the g weights of the elements
                    for e in p.get_elements():
                        if(e.name in g_weights.keys()):
                            g_weights[e.name] *= g_multiplier
                        else:
                            g_weights[e.name] = initial_g
                    break #exit the while

            # Generate children connections
            connections = []
            for _, element in possible_connections.iterrows():
                already_visited = any(e.name == element["name"] for e in closed_list)
                if (
                    already_visited or
                    not current_node.connectable(element, D) or # skip not connectable elements
                    (len(closed_list)==1 and element["name"] == end_node.name) # avoid a path customer-terminal (start node - end node)
                    ):
                    continue

                g = g_weights[element["name"]] if element["name"] in g_weights.keys() else initial_g
                # Create new node
                new_node = Node(current_node, element, g)
                connections.append(new_node)

            # Loop through connections
            for connection in connections:
                for closed_node in closed_list:
                    if connection == closed_node: # connection is on the closed list. 
                        continue

                # Calculate the g and h values
                d = current_node.distance(connection)
                temp_g = connection.g + current_node.g + d
                connection.h = current_node.distance(end_node)

                for open_node in open_list:
                    if (connection == open_node and #connection already inside the open list
                            temp_g < open_node.g): #the connection has a lower g then the node already present in the open list. Update its values, including the parent
                            open_node.g = temp_g
                            open_node.f = open_node.g + open_node.h
                            open_node.parent = current_node

                else:
                    # Add the child to the open list
                    connection.g = temp_g
                    connection.f = connection.g + connection.h
                    open_list.append(connection)

    return paths

def BuildHMatrix(h_paths, elements):
    cols = elements["name"].values
    H = pd.DataFrame(columns=cols)

    for i,h in enumerate(h_paths):
        matrix_path = [0] * len(cols)
        for e in h.get_elements_names():
            idx = int(e[1:])
            matrix_path[idx-1] = 1
        H.loc[f"h{i+1}"] = matrix_path
    return H

def MatrixDivision(elements, H, C,R,T):
    Hc = H.iloc[:,  0               :len(C)]
    Hr = H.iloc[:,  len(C)          :len(C)+len(R)]
    Ht = H.iloc[:,  len(C)+len(R)   :-len(Subset(elements, ["transformer"]))]
    return Hc, Hr, Ht


def optimization_problem(C,R,T, H,Hc,Hr,Ht):
    model = gp.Model()
    Hc, Hr, Ht = Hc.values, Hr.values, Ht.values

    MatrixTr_size = (len(T), len(R))
    MatrixP_size = len(H)

    # Decision Variables
    Tr = model.addMVar(MatrixTr_size, vtype=gp.GRB.BINARY, name="Tr")
    P = model.addMVar(MatrixP_size, vtype=gp.GRB.BINARY, name="P")

    # Objective function 22a
    model.setObjective( 10 * P.sum() - Tr.sum(), sense=gp.GRB.MAXIMIZE )

    #Constaint 22b
    for k in range(Tr.shape[0]):
        for m in range(MatrixP_size):
            model.addConstr( np.sum(Hr,axis=1)[m] * P[m] * np.transpose(Ht)[k,m] <= ( (Tr @ np.transpose(Hr)) * np.transpose(Ht))[k,m] )

    #Constaint 22c
    # for k in range(len(C)):
    model.addConstr( P @ Hc <= 1)

    #Constaint 22d
    for k in range(len(R)):
        model.addConstr( gp.quicksum(Tr)[k] <= 1)

    model.optimize()
    return model, Tr, P

def transform_matrices_in_dfs(H,T,R, Tr, P):
    Tr_sol = pd.DataFrame()
    for tr in Tr:
        tr_sol = []
        for r  in tr:
            tr_sol.append(int(r.X))
        Tr_sol = pd.concat([Tr_sol, pd.DataFrame(tr_sol)], axis=1, ignore_index=True)
    Tr_sol = Tr_sol.T
    Tr_sol.columns = list(R["name"].values)
    Tr_sol.index = list(T["name"].values)

    p_sol = []
    for p in P:
        p_sol.append(p.X)
    P_sol = pd.DataFrame(p_sol, dtype=np.int8).T
    P_sol.columns = list(H.index)

    return Tr_sol, P_sol

def DiagnosticFunction(H, C, P_sol, Tr_sol):
    number_issues = 0
    ### 1st Check (check on each customer having a path) ###
    print("### Cheking 1st condition: check on each customer having a path ###")
    estimated_optimal_paths = [p for p in P_sol.columns.values if P_sol[p][0] == 1]

    customers_wo_paths = set(C.name)
    for h in estimated_optimal_paths:
        elements = H.loc[h]
        path_customers = [elements.index[i] for i, e in enumerate(elements[:len(C)]) if e == 1]
        if path_customers:
            customers_wo_paths.discard(path_customers[0])

    if(customers_wo_paths):
        number_issues +=1
        for c in customers_wo_paths:
            print(f"For customer: {c} was not possible to identify a path")
    else:
        print("No issue identified. For each customers a path was identified")("No issue identified.")

    ### Final check ###
    print("\n\n### Summary ###")
    if(number_issues==0):
        print("No issue identified")
    else:
        print(f"{number_issues} issue(s) identified")
        
    return customers_wo_paths

def find_connections(elements, Tr_sol, P_sol, paths):
    for terminal, elems in Tr_sol.iterrows():
        for i,e in elems.items():
            if(e==1):
                ind = elements[elements["name"]==i].index[0]
                elements.at[ind, 'terminal'] = terminal

    connections = gpd.GeoDataFrame(columns=["from_id", "to_id", "terminal", "coor"])

    estimated_optimal_paths = [p for p in P_sol.columns.values if P_sol[p][0] == 1]
    pp = [p.get_elements_names() for p in paths]
    estimated_optimal_paths = [pp[int(i[1:])-1] for i in estimated_optimal_paths]
    for h in estimated_optimal_paths[:-1]:
        for i in range(len(h)-1):
            elem_from = elements[elements["name"]==h[i]]
            elem_to = elements[elements["name"]==h[i+1]]

            
            coor_from = elem_from.coor.values[0]
            coor_to = elem_to.coor.values[0]

            # Calculate nearest points (if necessary, otherwise use coordinates directly)
            nearest_point_from = nearest_points(coor_from, coor_to)[0]
            nearest_point_to = nearest_points(coor_to, coor_from)[0]

            l = LineString([nearest_point_from, nearest_point_to])

            new_row = {
                            'from_id': elem_from.name.values[0], 
                            'to_id': elem_to.name.values[0], 
                            'terminal': h[-1], 
                            'coor': l
                        }
            new_row = gpd.GeoDataFrame([new_row])
            connections = pd.concat([connections, new_row], ignore_index=True)
    connections = connections.set_geometry('coor')
    return connections