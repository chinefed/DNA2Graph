import cv2
import numpy as np
import networkx as nx
from skimage.draw import line


def get_uniform_grid_graph(G):
    '''
    Convert a graph with uneven spacing to a uniform grid graph.
    '''
    def get_line_coordinates(start, end):
        '''
        Get the coordinates of a line between two points.
        '''
        start_y, start_x = start
        end_y, end_x = end
        line_y, line_x = line(start_y, start_x, end_y, end_x)

        return list(zip(line_y, line_x))

    G_uniform_grid = nx.Graph()
    G_uniform_grid.graph['h'] = G.graph['h']
    G_uniform_grid.graph['w'] = G.graph['w']
    G_uniform_grid.graph['boundary_skel_pixels'] = G.graph['boundary_skel_pixels']

    for start, end in G.edges():
        line_coords = get_line_coordinates(start, end)
        for u, v in zip(line_coords[:-1], line_coords[1:]):
            G_uniform_grid.add_edge(u, v)

    return G_uniform_grid


def get_graph_bounding_box(G):
    '''
    Get the bounding box of a graph.
    The format is (y_min, x_min, y_max, x_max).
    '''
    nodes = list(G.nodes)
    y_coords, x_coords = zip(*nodes)

    y_min = min(y_coords)
    x_min = min(x_coords)
    y_max = max(y_coords) + 1
    x_max = max(x_coords) + 1

    return (y_min, x_min, y_max, x_max)


def linear_walk(G, start, v, max_steps=None):
    '''
    Perform a walk on a graph starting from `start` and proceeding through 
    nodes of degree 2, beginning with neighbor `v`.

    The walk continues by traversing the path of degree-2 nodes in a straight 
    line (i.e., always choosing the neighbor that is not the previous node),
    and stops when a node with degree not equal to 2 is encountered or when
    v == start or when `max_steps` is reached.
    '''
    if start not in list(G.neighbors(v)):
        raise ValueError(
            f"Node '{v}' is not a neighbor of node '{start}'."
        )
    
    if max_steps is not None and (max_steps < 1 or not isinstance(max_steps, int)):
        raise ValueError(f"'max_steps' must be a positive integer or None. Received: {max_steps}.")

    # If not specified set max_steps to infinity
    max_steps = np.inf if max_steps is None else max_steps

    n = 1
    walk = [start, v] # Sequence of visited nodes initialized with first step
    while G.degree(v) == 2 and v != start and n < max_steps:
        # We know v has degree 2
        nbr1, nbr2 = list(G.neighbors(v))
        # Move in the opposite direction with respect to the one you come from
        parent = walk[-2]
        v = nbr1 if nbr2 == parent else nbr2
        walk.append(v)
        n += 1

    # Check if the walk terminated because the maximum number of steps was reached
    max_steps_reached = G.degree(v) == 2 and v != start and n == max_steps

    return walk, max_steps_reached


def linear_walk_decomposition(G):
    '''
    Decompose a graph into a collection of linear walks.
    '''
    nodes_to_visit = set(G.nodes)
    blocked_edges = set()
    walks = []

    # Get list of endpoint (degree 1 nodes) and junctions (degree > 2 nodes)
    breakpoints = [u for u in G.nodes if G.degree[u] != 2]

    # Start linear walks from "breakpoints" along each incident edge
    for u in breakpoints:
        for v in G.neighbors(u):
            edge = frozenset((u, v))
            if edge in blocked_edges:
                continue
            walk, _ = linear_walk(G, u, v)
            walks.append(walk)

            # Mark last edge as blocked to prevent walking along
            # the same path from the opposite direction
            edge_last = frozenset((walk[-2], walk[-1]))
            blocked_edges.add(edge_last)

            # Mark nodes along the walk as visited
            nodes_to_visit.difference_update(walk)

    while nodes_to_visit:
        # Unvisited nodes are part of pure cycles or isolated nodes
        u = next(iter(nodes_to_visit)) # Get an arbitrary unvisited node

        # Get neighbors of u
        neighbors = list(G.neighbors(u))
        if not neighbors:
            # Isolated node
            nodes_to_visit.remove(u)
            continue

        v = neighbors[0]
        walk, _ = linear_walk(G, u, v)
        walks.append(walk)

        # Mark nodes along the walk as visited
        nodes_to_visit.difference_update(walk)

    return walks


def skeleton_to_graph(skeleton):
    '''
    Converts a binary skeleton image to a graph representation.
    '''
    h, w = skeleton.shape
    G = nx.Graph()

    # Store h and w in the graph attributes
    G.graph['h'] = h
    G.graph['w'] = w

    # Add the nodes for each pixel in the skeleton
    coords = np.argwhere(skeleton)
    for y, x in coords:
        G.add_node((y, x), dist=skeleton[y, x])

    # Add the edges based on 8-connectivity
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                yi, xi = y + dy, x + dx
                if 0 <= yi < h and 0 <= xi < w and skeleton[yi, xi]:
                    G.add_edge((y, x), (yi, xi))
    return G


def graph_to_skeleton(G):
    '''
    Convert a graph representation back to a binary skeleton image.
    '''
    h = G.graph['h']
    w = G.graph['w']

    skeleton = np.full((h, w), False)
    coords = np.array(list(G.nodes), dtype=np.int32)
    rows, cols = coords[:, 0], coords[:, 1]
    skeleton[rows, cols] = True

    return skeleton


def graph_to_mask(G, dilation_kernel_size=5):
    '''
    Convert a graph to a binary mask.
    '''
    # Convert to a uniform grid graph (8-connected nodes on the pixel grid)
    G = get_uniform_grid_graph(G)
    
    # Convert the graph to a skeleton
    skeleton = graph_to_skeleton(G).astype(np.uint8) * 255

    # Dilate the skeleton to create a mask
    dilation_kernel = np.ones(
        (dilation_kernel_size, dilation_kernel_size),
        np.uint8
    )
    mask = cv2.dilate(skeleton, dilation_kernel, iterations=1)

    return mask


def mark_boundary_components(G, boundary_skeleton_pixels):
    '''
    Identify and mark components in the graph with endpoints
    touching the boundary of valid regions in the image representation.
    These components are likely to represent incomplete or partial structures,
    which may not be fully captured within the valid imaging area.
    Incomplete structures may bias measurements or interpretations
    in downstream analysis.
    '''
    G = G.copy()

    # Initialize all nodes as non-boundary
    nx.set_node_attributes(G, False, 'is_boundary')

    for pixel in boundary_skeleton_pixels:
        if pixel not in G or G.nodes[pixel]['is_boundary'] or G.degree(pixel) != 1:
            # Skip a pixel if:
            # - it does not correspond to a node in the graph representation
            # - the corresponding node has already been marked as boundary
            # - the corresponding node is not an endpoint (degree != 1)
            continue

        # Retrieve the connected component that contains the graph node
        # corresponding to the boundary skeleton pixel
        boundary_component = nx.node_connected_component(G, pixel)

        # Mark all nodes in this component as boundary
        nx.set_node_attributes(G, {v: True for v in boundary_component}, 'is_boundary')

    return G