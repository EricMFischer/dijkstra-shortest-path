## Synopsis
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
integers.

IMPLEMENTATION NOTES: This graph is small enough that the straightforward O(mn) time
implementation of Dijkstra's algorithm should work fine. OPTIONAL: For those of you seeking an
additional challenge, try implementing the heap-based version. Note this requires a heap that
supports deletions, and you'll probably need to maintain some kind of mapping between vertices and
their positions in the heap.

## Motivation

Dijkstra's shortest path algorithm, which solves from the shortest path for a given vertex to any other one, demonstrates how many algorithms can be optimized with a data structure, in this case a binary heap. The straightforward solution has **O(mn)** time complexity, **n** for the number of loop iterations through vertices and **m** for the worst-case scenario of performing a linear scan through all the edges. The heap implementation has an **O(mlogn)** time complexity, **log(n)** for the height of the binary tree representation of a heap and **m** again for a linear scan through the edges.

(Note how the "big-oh" time complexity is lower-bounded by the number of edges and not vertices, as edges will always outnumber vertices in an undirected graph.) To summarize, any algorithm which calls for repeated minimum computations, like Dijkstra's shortest path algorithm, calls for a heap optimization.  

## Acknowledgements

This algorithm is part of the Stanford University Algorithms 4-Course Specialization on Coursera, instructed by Tim Roughgarden.
