from collections import deque
from collections.abc import Iterable, Mapping, Sequence, Set

from funmaze.graph import IGraph, Node, graph_neighbours


# https://www.geeksforgeeks.org/print-paths-given-source-destination-using-bfs/
# https://stackoverflow.com/a/64667117/2863746
def solve_bfs_paths(graph: IGraph[Node], start: Node, end: Node,
                    allow_cycles: bool = True,
                    ) -> Iterable[Sequence[Node]]:
    """Find all paths on the *graph* from *start* to *end*
    by breadth-first-search.

    If your graph has cycles, set *allow_cycles* to ``False`` if you want the
    iterator not to consider these paths (if you do, the iterator will never
    stop, because it will keep visiting cycles).
    Checking for cycles causes some slowing down, so they are allowed by
    default.

    .. warning:: Implementation consumes a lot of memory.
    """
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    queue: deque[list[Node]] = deque()
    queue.append([start])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            yield path
        else:
            for node2 in neighbours[node]:
                if allow_cycles or node2 not in path:
                    queue.append(path + [node2])


# https://en.wikipedia.org/wiki/Breadth-first_search#Pseudocode
# https://www.baeldung.com/cs/graph-algorithms-bfs-dijkstra
def solve_bfs_one_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Sequence[Node] | None:
    """Use a breadth-first-search on the graph to find one shortest path
    between two nodes.

    This implementation is almost identical to Dijkstra's algorithm
    where we exploit that the distance along every edge is equal to one
    to make for a slightly faster algorithm.
    The difference is that, unlike Dijkstra, we do not need to use a priority
    queue ordered by distance, and instead can use a simple FIFO queue,
    since this will be ordered by distance automatically (as edges have the
    same distance).
    """
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    parent: dict[Node, Node] = {}

    def _backtrack(node3) -> Sequence[Node]:
        path: deque[Node] = deque([node3])
        while (parent_node := parent[node3]) != start:
            node3 = parent_node
            path.appendleft(node3)
        path.appendleft(start)
        return path

    visited: set[Node] = {start}
    queue: deque[Node] = deque()
    queue.append(start)
    while queue:
        node = queue.popleft()
        if node == end:
            return _backtrack(node)
        else:
            for node2 in neighbours[node]:
                if node2 not in visited:
                    visited.add(node2)
                    parent[node2] = node
                    queue.append(node2)
    return None


# https://stackoverflow.com/a/14145564/2863746 "bfs + reverse dfs"
def solve_bfs_all_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> IGraph[Node]:
    """Use a breadth-first-search on the graph to find a subgraph representing
    all shortest paths between two nodes.

    To list the actual paths, use :func:`solve_bfs_paths` on the subgraph.

    This implementation does a forward bfs to find the distance from *start*
    to every other node in the graph until *end* is reached.
    The result is returned as a graph, such that all paths from *start*
    to *end* on this graph are the shortest paths.
    """
    neighbours = graph_neighbours(graph)
    distances: dict[Node, int] = {start: 0}
    queue: deque[tuple[Node, int]] = deque([(start, 0)])
    while queue:
        node, distance = queue.popleft()
        if node == end:
            break
        else:
            distance2 = distance + 1
            for node2 in neighbours[node]:
                # ever visited?
                if node2 not in distances:
                    queue.append((node2, distance2))
                # set distance if not yet set, return edge if distance correct
                if distances.setdefault(node2, distance2) == distance2:
                    yield node, node2
