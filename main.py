import time
from collections import deque
import heapq


# 1. ESTRUCTURA Y CLASES BASE DE AIMA (Problem y Node)

class Problem:
    """Clase base para definir un problema de busqueda (AIMA)."""
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return state in self.goal
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1


class Node:
    """Clase Nodo para construir el arbol de busqueda (AIMA)."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, node):
        return self.path_cost < node.path_cost

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
        return Node(next_state, self, action, next_cost)

    def path_tree(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


# 2. DEFINICION DEL PROBLEMA: AGENTE ORGANIZADOR DE ARCHIVOS

class FileOrganizerProblem(Problem):
    """
    Estado: tupla (Estado_Directorio, Duplicados_MB, Formato_Nombres)
        - Estado_Directorio: 'Desordenado', 'Limpio', 'Clasificado', 'Organizado'
        - Duplicados_MB: Espacio ocupado por archivos duplicados/basura (150 MB -> 0 MB)
        - Formato_Nombres: 'SinFormato', 'Estandarizado', 'Verificado'
    """
    def __init__(self, initial=('Desordenado', 150, 'SinFormato'),
                 goal=('Organizado', 0, 'Verificado')):
        super().__init__(initial, goal)

    def actions(self, state):
        directorio, dup_mb, formato = state
        accs = []
        if directorio == 'Desordenado':
            accs.append('Eliminar_Duplicados')
            accs.append('Mover_Sin_Limpiar')
        elif directorio == 'Limpio':
            accs.append('Renombrar_Lote')
        elif directorio == 'Clasificado' and formato == 'SinFormato':
            accs.append('Limpiar_Y_Renombrar_En_Destino')
        elif directorio == 'Clasificado' and formato == 'Estandarizado':
            accs.append('Verificar_Integridad')
        return accs

    def result(self, state, action):
        directorio, dup_mb, formato = state

        if action == 'Eliminar_Duplicados':
            return ('Limpio', 0, 'SinFormato')
        elif action == 'Renombrar_Lote':
            return ('Clasificado', 0, 'Estandarizado')
        elif action == 'Mover_Sin_Limpiar':
            return ('Clasificado', dup_mb, 'SinFormato')
        elif action == 'Limpiar_Y_Renombrar_En_Destino':
            return ('Clasificado', 0, 'Estandarizado')
        elif action == 'Verificar_Integridad':
            return ('Organizado', 0, 'Verificado')

        return state

    def goal_test(self, state):
        directorio, dup_mb, formato = state
        return directorio == 'Organizado' and dup_mb == 0 and formato == 'Verificado'

    def path_cost(self, c, state1, action, state2):
        costs = {
            'Eliminar_Duplicados': 4,             # Borrar basura en origen (bajo costo)
            'Renombrar_Lote': 6,                  # Renombrar y clasificar archivos limpios
            'Mover_Sin_Limpiar': 15,              # Penalizacion por trasladar 150 MB de duplicados
            'Limpiar_Y_Renombrar_En_Destino': 12, # Alto costo por reindexar subdirectorios
            'Verificar_Integridad': 2             # Chequeo final del directorio
        }
        return c + costs.get(action, 1)


# Heurística admisible para A* (costo mínimo estimado restante)
def heuristica_organizador(node):
    directorio, dup_mb, formato = node.state
    h = 0
    if directorio == 'Desordenado':
        h += 12  # Estimacion admisible: Eliminar (4) + Renombrar (6) + Verificar (2) = 12
    elif directorio == 'Limpio':
        h += 8   # Estimacion admisible: Renombrar (6) + Verificar (2) = 8
    elif directorio == 'Clasificado' and formato == 'SinFormato':
        h += 14  # Estimacion admisible: Limpiar_Destino (12) + Verificar (2) = 14
    elif directorio == 'Clasificado' and formato == 'Estandarizado':
        h += 2   # Estimacion admisible: Verificar (2)
    return h


# 3. ALGORITMOS DE BUSQUEDA (BFS, DFS, A*)

def breadth_first_graph_search(problem):
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node, 0
    frontier = deque([node])
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child, nodes_expanded
                frontier.append(child)
    return None, nodes_expanded


def depth_first_graph_search(problem):
    frontier = [Node(problem.initial)]
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node, nodes_expanded
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
    return None, nodes_expanded


def astar_search(problem, h=heuristica_organizador):
    node = Node(problem.initial)
    frontier = []
    heapq.heappush(frontier, (node.path_cost + h(node), node))
    explored = set()
    nodes_expanded = 0

    while frontier:
        f, node = heapq.heappop(frontier)
        if problem.goal_test(node.state):
            return node, nodes_expanded
        if node.state not in explored:
            explored.add(node.state)
            nodes_expanded += 1
            for child in node.expand(problem):
                if child.state not in explored:
                    heapq.heappush(frontier, (child.path_cost + h(child), child))
    return None, nodes_expanded


# 4. EJECUCION Y TABLA DE RESULTADOS

if __name__ == '__main__':
    problem = FileOrganizerProblem()
    algoritmos = [
        ("BFS (Anchura)", breadth_first_graph_search),
        ("DFS (Profundidad)", depth_first_graph_search),
        ("A* (Estrella)", astar_search)
    ]

    print("\n" + "=" * 80)
    print(f"{'Algoritmo':<20} | {'Nodos Exp.':<12} | {'Tiempo (ms)':<12} | {'Costo I/O':<15} | {'Óptimo':<8}")
    print("=" * 80)

    for nombre, func in algoritmos:
        t0 = time.perf_counter()
        sol_node, exp_nodes = func(problem)
        t1 = time.perf_counter()
        
        tiempo_ms = (t1 - t0) * 1000
        costo = sol_node.path_cost if sol_node else "N/A"
        es_optimo = "Sí" if costo == 12 else "No"

        print(f"{nombre:<20} | {exp_nodes:<12} | {tiempo_ms:<12.4f} | {costo:<15} | {es_optimo:<8}")

    print("=" * 80 + "\n")
    
    if sol_node:
        print("Camino óptimo encontrado por A*:")
        for i, step in enumerate(sol_node.path_tree()):
            print(f"  Paso {i}: Estado = {step.state} | Acción = {step.action} | Costo Acumulado = {step.path_cost}")
        print("\n")