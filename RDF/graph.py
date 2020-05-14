from collections import defaultdict
import re

class Vertex(object):
    vertex_counter = 0
    
    def __init__(self, name):
        self.name = name
        self.id = Vertex.vertex_counter
        Vertex.vertex_counter += 1
        
    def __eq__(self, other):
        if other is None: 
            return False
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

class Edge(object):
    edge_counter = 0
    
    def __init__(self, name, _from=None, _to=None):
        self.name = name
        # self._from = _from
        # self._to = _to
        self.id = Edge.edge_counter
        Edge.edge_counter += 1
        
    def __eq__(self, other):
        if other is None: 
            return False
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

class KnowledgeGraph(object):
    def __init__(self):
      self._vertice_map = {}
      self._edge_map = {}
      self._vertices = set()
      self._transition_matrix = defaultdict(set)
      self._inv_transition_matrix = defaultdict(set)
        
    def add_vertex(self, vertex_str):
        """Add a vertex to the Knowledge Graph."""
        vertex = None
        if vertex_str in self._vertice_map:
          vertex = self._vertice_map[vertex_str]
        elif vertex_str == '':
          vertex = None
        else:
          vertex = Vertex(vertex_str)
          self._vertice_map[vertex_str] = vertex
        
        self._vertices.add(vertex)
        
        return vertex

    def add_edge(self, v1, edge_str, v2):
        """Add a uni-directional edge."""
        edge = None
        if edge_str in self._edge_map:
          edge = self._edge_map[edge_str]
        elif edge_str == None or edge_str=='':
          edge = None
        else:
          edge = Edge(edge_str)
          self._edge_map[edge_str] = edge
        
        self._transition_matrix[v1].add((v2, edge))
        self._inv_transition_matrix[v2].add((v1, edge))
       
        
    # def remove_edge(self, v1, v2):
    #     """Remove the edge v1 -> v2 if present."""
    #     if v2 in self._transition_matrix[v1]:
    #         self._transition_matrix[v1].remove(v2)

    def get_neighbors(self, vertex):
        """Get all the neighbors of vertex (vertex -> neighbor)."""
        return self._transition_matrix[vertex]

    def get_inv_neighbors(self, vertex):
        """Get all the neighbors of vertex (vertex -> neighbor)."""
        return self._inv_transition_matrix[vertex]
    
    def visualise(self):
        """Visualise the graph using networkx & matplotlib."""
        import matplotlib.pyplot as plt
        import networkx as nx
        nx_graph = nx.DiGraph()
        
        for v in self._vertices:
            if not v.predicate:
                name = v.name.split('/')[-1]
                nx_graph.add_node(name, name=name, pred=v.predicate)
            
        for v in self._vertices:
            if not v.predicate:
                v_name = v.name.split('/')[-1]
                # Neighbors are predicates
                for pred in self.get_neighbors(v):
                    pred_name = pred.name.split('/')[-1]
                    for obj in self.get_neighbors(pred):
                        obj_name = obj.name.split('/')[-1]
                        nx_graph.add_edge(v_name, obj_name, name=pred_name)
        
        plt.figure(figsize=(10,10))
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        names = nx.get_edge_attributes(nx_graph, 'name')
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names)
