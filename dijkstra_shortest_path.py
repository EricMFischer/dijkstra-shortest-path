'''
The file contains an adjacency list representation of an undirected weighted graph with 200
vertices labeled 1 to 200. Each row consists of the node tuples that are adjacent to that
particular vertex along with the length of that edge. For example, the 6th row has 6 as the first
entry indicating that this row corresponds to the vertex labeled 6. The next entry of this row
"141,8200" indicates that there is an edge between vertex 6 and vertex 141 that has length 8200.
The rest of the pairs of this row indicate the other vertices adjacent to vertex 6 and the lengths
of the corresponding edges.

Your task is to run Dijkstra's shortest-path algorithm on this graph, using 1 (the first vertex)
as the source vertex, and to compute the shortest-path distances between 1 and every other vertex
of the graph. If there is no path between a vertex vv and vertex 1, we'll define the shortest-path
distance between 1 and vv to be 1000000.

You should report the shortest-path distances to the following ten vertices, in order:
7,37,59,82,99,115,133,165,188,197. You should encode the distances as a comma-separated string of
integers. So if you find that all ten of these vertices except 115 are at distance 1000 away from
vertex 1 and 115 is 2000 distance away, then your answer should be
1000,1000,1000,1000,1000,2000,1000,1000,1000,1000. Remember the order of reporting DOES MATTER.

IMPLEMENTATION NOTES: This graph is small enough that the straightforward O(mn) time
implementation of Dijkstra's algorithm should work fine. OPTIONAL: For those of you seeking an
additional challenge, try implementing the heap-based version. Note this requires a heap that
supports deletions, and you'll probably need to maintain some kind of mapping between vertices and
their positions in the heap.
'''
import pprint
import time


# input: file name
# output: object with vertex keys and their neighbors with weights
# e.g. ['2,20', '3,12']
def preprocess_adj_list(filename):
    graph = {}
    with open(filename) as f_handle:
        for line in f_handle:
            v_arr = line.split()
            k = v_arr[0]
            graph[k] = v_arr[1:]
    return graph


# input: object with vertex keys and their neighbors with weights
# output: Graph instantiated with input graph object
def create_graph(graph_obj):
    G = Graph()
    for v_key in graph_obj:
        v = Vertex(int(v_key))
        for nbr in graph_obj[v_key]:
            w_key, edge = nbr.split(',')
            v.add_nbr(int(w_key), int(edge))
        G.add_v(v)
    return G


# Vertex class for undirected graphs
class Vertex(object):
    def __init__(self, key):
        self.key = key
        self.nbrs = {}

    def __str__(self):
        return '{' + "'key': {}, 'nbrs': {}".format(
            self.key,
            self.nbrs
        ) + '}'

    def add_nbr(self, nbr_key, weight=1):
        if (nbr_key):
            self.nbrs[nbr_key] = weight

    def has_nbr(self, nbr_key):
        return nbr_key in self.nbrs

    def get_nbr_keys(self):
        return list(self.nbrs.keys())

    def remove_nbr(self, nbr_key):
        if nbr_key in self.nbrs:
            del self.nbrs[nbr_key]

    def get_edge(self, nbr_key):
        if nbr_key in self.nbrs:
            return self.nbrs[nbr_key]


# Undirected graph class
# Note: to maximize applications, add_edge, increase_edge, and remove_edge only add or remove an
# edge for the 'from' vertex, and has_edge only checks the 'from' vertex.
class Graph(object):
    def __init__(self):
        self.vertices = {}

    # 'x in graph' will use this containment logic
    def __contains__(self, key):
        return key in self.vertices

    # 'for x in graph' will use this iter() definition, where x is a vertex in an array
    def __iter__(self):
        return iter(self.vertices.values())

    def __str__(self):
        output = '\n{\n'
        vertices = self.vertices.values()
        for v in vertices:
            graph_key = "{}".format(v.key)
            v_str = "\n   'key': {}, \n   'nbrs': {}".format(
                v.key,
                v.nbrs
            )
            output += ' ' + graph_key + ': {' + v_str + '\n },\n'
        return output + '}'

    def add_v(self, v):
        if v:
            self.vertices[v.key] = v
        return self

    def get_v(self, key):
        try:
            return self.vertices[key]
        except KeyError:
            return None

    def get_v_keys(self):
        return list(self.vertices.keys())

    # removes vertex as neighbor from all its neighbors, then deletes vertex
    def remove_v(self, key):
        if key in self.vertices:
            nbr_keys = self.vertices[key].get_nbr_keys()
            for nbr_key in nbr_keys:
                self.remove_edge(nbr_key, key)
            del self.vertices[key]
        return self

    def add_edge(self, from_key, to_key, weight=1):
        if from_key not in self.vertices:
            self.add_v(Vertex(from_key))
        if to_key not in self.vertices:
            self.add_v(Vertex(to_key))

        self.vertices[from_key].add_nbr(to_key, weight)

    # adds the weight for an edge if it exists already, with a default of 1
    def increase_edge(self, from_key, to_key, weight=1):
        if from_key not in self.vertices:
            self.add_v(Vertex(from_key))
        if to_key not in self.vertices:
            self.add_v(Vertex(to_key))

        weight_u_v = self.get_v(from_key).get_edge(to_key)
        new_weight_u_v = weight_u_v + weight if weight_u_v else weight

        self.vertices[from_key].add_nbr(to_key, new_weight_u_v)
        return self

    def has_edge(self, from_key, to_key):
        if from_key in self.vertices:
            return self.vertices[from_key].has_nbr(to_key)

    def remove_edge(self, from_key, to_key):
        if from_key in self.vertices:
            self.vertices[from_key].remove_nbr(to_key)

    def for_each_v(self, cb):
        for v in self.vertices:
            cb(v)


# input:
# output:
class Heap():
    return {}


# input: Graph, source vertex key, and vertices to which to find a shortest path
# output: shortest paths from source to input vertices
def dijkstra_shortest_path(G, source_key, vertices):
    X = {}  # vertices explored so far
    A = {}  # shortest path distances from source vertex 1
    X[source_key] = 1
    A[source_key] = 0

    G_keys_len = len(G.get_v_keys())
    while len(X.keys()) is not G_keys_len:
        # records shortest path for every explored vertex to one of its unexplored neighbors
        # Ex: {100: [2,3]} -> shortest path from vertex 2 is to vertex 3 with a distance of 100
        shortest_paths = {}

        for v_key in X:
            v = G.get_v(v_key)
            nbr_keys = list(filter(lambda k: k not in X, v.get_nbr_keys()))

            nbr_paths = {}
            for nbr_key in nbr_keys:
                # local shortest path
                nbr_paths[nbr_key] = A[v_key] + v.get_edge(nbr_key)

            if nbr_paths:
                min_nbr_key = min(nbr_paths, key=nbr_paths.get)
                # record shortest path for v_key, e.g. 2
                shortest_paths[nbr_paths[min_nbr_key]] = [v_key, min_nbr_key]  # {100: [2,3]}

        # book-keeping
        v_key, w_key = shortest_paths[min(shortest_paths.keys())]
        X[w_key] = 1
        # In this implementation, only set shortest path after checking every explored to
        # unexplored vertex, i.e. will definitely be shortest path
        # With heap, we recalculate shortest paths to unexplored vertices stored as keys, so
        # each time we add a vertex to X and remove it from heap, we can simply call extract_min
        # to get shortest path key
        A[w_key] = A[v_key] + G.get_v(v_key).get_edge(w_key)  # shortest path from source to w

    result = []
    for v_key in vertices:
        path = A[v_key] if A[v_key] else 1000000
        result.append(path)
    return result


def main():
    graph_obj = preprocess_adj_list('dijkstra_shortest_path.txt')
    # pprint.pprint(graph_obj, width=40)
    G = create_graph(graph_obj)
    # print(G)

    start = time.time()
    result = dijkstra_shortest_path(G, 1, [7, 37, 59, 82, 99, 115, 133, 165, 188, 197])
    print('result: ', result)
    print('elapsed time: ', time.time() - start)


main()
